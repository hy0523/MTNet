import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import importlib
from core.manager.engine import MTNetEngine
from core.dataloader.video_vos_dataset import VOSDataset
from core.dataloader.davis_test_dataset import DAVISTestDataset
import math
from core.learning.learning import adjust_learning_rate
from core.measures.jaccard import db_eval_iou_multi
from core.utils.distributed_utils import is_main_process
from core.utils.image_saver import im_transform, mask_transform
import cv2 as cv
import datetime


class Trainer(object):
    def __init__(self, config, logger=None, writer=None, local_rank=0):
        self.config = config
        self.logger = logger
        self.writer = writer
        self.rank = local_rank
        self.log_text_interval = config['log_text_interval']
        self.log_image_interval = config['log_image_interval']
        self.do_log = config['do_log']
        if config['amp']:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        engine_config = importlib.import_module('core.networks.' + config['model_name'])
        if config['distributed']:
            model = engine_config.MTNet(config).cuda(local_rank)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(local_rank)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[local_rank],
                                                              output_device=local_rank)
        else:
            model = engine_config.MTNet(config).cuda()

        if config['distributed']:
            backbone_params = list(map(id, model.module.backbone.parameters()))
            other_params = filter(lambda p: id(p) not in backbone_params, model.module.parameters())
            backbone_params = filter(lambda p: id(p) in backbone_params, model.module.parameters())
        else:
            backbone_params = list(map(id, model.backbone.parameters()))
            other_params = filter(lambda p: id(p) not in backbone_params, model.parameters())
            backbone_params = filter(lambda p: id(p) in backbone_params, model.parameters())
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': config['lr']},
            {'params': other_params, 'lr': config['lr']}
        ], lr=config['lr'], weight_decay=config['adamw_weight_decay'])
        self.engine = MTNetEngine(model, optimizer, config)
        self.criterion = nn.BCEWithLogitsLoss()

    def prepare_dataset(self):
        config = self.config
        stage = config['stage']
        assert stage in ['pre_train', 'main_train']
        if stage == 'pre_train':
            train_dataset = VOSDataset(im_root="../../CLIPVOS/data/YouTube2SEG_CLIP/train/frame",
                                       fl_root="../../CLIPVOS/data/YouTube2SEG_CLIP/train/flow",
                                       gt_root="../../CLIPVOS/data/YouTube2SEG_CLIP/train/mask")
        else:
            train_dataset = VOSDataset(im_root="../../CLIPVOS/data/DAVIS2SEG_CLIP/train/frame",
                                       fl_root="../../CLIPVOS/data/DAVIS2SEG_CLIP/train/flow",
                                       gt_root="../../CLIPVOS/data/DAVIS2SEG_CLIP/train/mask", )

        val_dataset = DAVISTestDataset(im_root="../../CLIPVOS/data/DAVIS2SEG_CLIP/val/frame",
                                       fl_root="../../CLIPVOS/data/DAVIS2SEG_CLIP/val/flow",
                                       gt_root="../../CLIPVOS/data/DAVIS2SEG_CLIP/val/mask")
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if config[
            'distributed'] else None
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=config['batch_size'],
                                                        shuffle=False if config['distributed'] else True,
                                                        num_workers=config['num_workers'],
                                                        drop_last=True,
                                                        sampler=self.train_sampler,
                                                        pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=config['num_workers'],
                                                      drop_last=False,
                                                      sampler=None)

    def process_pretrained_model(self):
        config = self.config
        benchmark_model_dir = os.path.join(config['exp'], config['save_path'])
        self.epoch_resume = 0
        self.best_iou = 0.0
        self.curr_iter = 0
        # pre-train
        if config['pretrain_model_path'] != None:
            path = config['pretrain_model_path']
            self.engine.load_network(path)
            return
        # resume training
        if len(os.listdir(benchmark_model_dir)) > 2:
            checkpoint_names = os.listdir(benchmark_model_dir)
            max_number = 0
            best_iou = 0.0
            last_checkpoint_path = None
            for name in checkpoint_names:
                if name.split('_')[0].isdigit():
                    if int(name.split('_')[0]) > max_number:
                        max_number = int(name.split('_')[0])
                        last_checkpoint_path = name
            select_list = []
            for name in checkpoint_names:
                if 'best_' in name:
                    select_list.append(name)
            for name in select_list:
                if not name.split('_')[0].isdigit() and len(name) > 15:
                    if int(name.split('_')[1]) < max_number:
                        if float(name.split('_')[2][:-4]) > best_iou:
                            best_iou = float(name.split('_')[2][:-4])
            self.best_iou = best_iou
            last_checkpoint_path = os.path.join(benchmark_model_dir, last_checkpoint_path)
            curr_iter = self.engine.load_model(last_checkpoint_path, scaler=self.scaler)
            epoch_resume = math.ceil(curr_iter / len(self.train_loader))
            self.curr_iter = curr_iter
            self.epoch_resume = epoch_resume

    def clip_training(self):
        self.prepare_dataset()
        self.process_pretrained_model()
        config = self.config
        curr_iter = self.curr_iter
        epoch = self.epoch_resume
        best_iou = self.best_iou
        train_sampler = self.train_sampler
        train_loader = self.train_loader
        criterion = self.criterion
        engine = self.engine
        optimizer = engine.optimizer
        logger = self.logger
        engine.vos_model.train()
        start_time = time.time()
        while (curr_iter < config['max_iter']):
            if config['distributed']:
                train_sampler.set_epoch(epoch)
            epoch += 1
            for i, sample in enumerate(train_loader):
                if curr_iter > config['max_iter']:
                    break
                start = time.time()
                if config['stage'] == 'pre_train':
                    now_lr = adjust_learning_rate(
                        optimizer=optimizer,
                        base_lr=config['lr'],
                        p=config['power'],
                        itr=curr_iter,
                        restart=1,
                        warm_up_steps=config['warm_up_steps'],
                        warmup_lr_start=1e-6,
                        is_cosine_decay=False,
                        min_lr=0.,
                        encoder_lr_ratio=1.,
                        max_itr=config['max_iter'])
                else:
                    now_lr = optimizer.param_groups[0]['lr']
                images = sample['rgb'].cuda()
                flows = sample['flow'].cuda()
                masks = sample['gt'].cuda().flatten(0, 1)
                video_name = sample['info']['name'][0]
                with torch.cuda.amp.autocast(enabled=config['amp']):
                    main_pred, aux1_pred, aux2_pred, aux3_pred = engine.vos_model(images, flows)
                    main_loss = criterion(main_pred, masks)
                    aux_loss_3 = criterion(aux3_pred, masks)
                    aux_loss_2 = criterion(aux2_pred, masks)
                    aux_loss_1 = criterion(aux1_pred, masks)
                    aux_loss = (aux_loss_1 + aux_loss_2 + aux_loss_3) * 0.5
                    loss = main_loss + aux_loss
                    iou = db_eval_iou_multi(masks.cpu().detach().numpy(), main_pred.sigmoid().cpu().detach().numpy())
                    loss = torch.mean(loss)
                optimizer.zero_grad(set_to_none=True)
                if self.config['amp']:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if (curr_iter + 1) % config['print_every'] == 0:
                    if is_main_process():
                        this_time = time.time() - start
                        logger.info(
                            'Video:{}\t Iter: [{}/{}]\tB-Lr {:.6f}\tH-Lr {:.6f}\tTime {:.3f}s\tTotal Loss: {:.4f}\tMain Loss: {:.4f}\tAux Loss: {:.4f}\tIOU: {:.4f}'.format(
                                video_name, curr_iter + 1, config['max_iter'], now_lr,
                                now_lr, this_time, loss.data.item(), main_loss.data.item(), aux_loss.data.item(), iou))
                curr_iter = curr_iter + 1
                if curr_iter % config['log_text_interval'] == 0 and curr_iter != 0:
                    if self.do_log:
                        if is_main_process():
                            self.writer.log_scalar('train/lr', now_lr, curr_iter)
                            self.writer.log_scalar('train/total_loss', loss.item(), curr_iter)
                            self.writer.log_scalar('train/main_loss', main_loss.item(), curr_iter)
                            self.writer.log_scalar('train/aux_loss', aux_loss.item(), curr_iter)
                            self.writer.log_scalar('train/iou', iou, curr_iter)
                if curr_iter % config['log_image_interval'] == 0 and curr_iter != 0:
                    if self.do_log:
                        img_list = []
                        flow_list = []
                        gt_list = []
                        pred_list = []
                        for i in range(config['train_frames']):
                            img = im_transform(images[0, i], size=(512, 512))
                            flow = im_transform(flows[0, i], size=(512, 512))
                            gt = mask_transform(masks[i, 0], size=(512, 512))
                            pred = mask_transform(main_pred[i, 0], size=(512, 512))
                            img_list.append(img)
                            flow_list.append(flow)
                            gt_list.append(gt)
                            pred_list.append(pred)
                        img_cat = cv.hconcat(img_list)
                        flow_cat = cv.hconcat(flow_list)
                        gt_cat = cv.hconcat(gt_list)
                        pred_cat = cv.hconcat(pred_list)
                        gt_cat = gt_cat[:, :, None].repeat(3, 2)
                        pred_cat = pred_cat[:, :, None].repeat(3, 2)
                        results = cv.vconcat([img_cat, flow_cat, gt_cat, pred_cat])
                        results = (results * 255).astype(np.uint8)
                        if is_main_process():
                            self.writer.log_cv2('train/pairs', results, curr_iter)
                # Save checkpoint
                if curr_iter % config['save_checkpoint_interval'] == 0 and curr_iter != 0:
                    engine.save_checkpoint(it=curr_iter, optimizer=optimizer, scaler=self.scaler)
                if config['eval'] and curr_iter % config['eval_iter'] == 0 and curr_iter > 0:
                    with torch.cuda.amp.autocast(enabled=config['amp']):
                        val_miou = self.clip_eval(curr_iter, engine, logger)
                    # Save model
                    if val_miou > best_iou:
                        if is_main_process():
                            logger.info('=' * 20)
                            logger.info('update: {} '.format(val_miou))
                            logger.info('=' * 20)
                            best_iou = val_miou
                            engine.save_model(curr_iter, is_best=True, best_iou=best_iou)
                    del val_miou
                    engine.vos_model.train()
        total_time = time.time() - start_time
        t_m, t_s = divmod(total_time, 60)
        t_h, t_m = divmod(t_m, 60)
        total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))
        if is_main_process():
            logger.info("Training End!")
            logger.info(f"Best MIoU:{best_iou}")
            logger.info(f"Total running time: {total_time}")
            logger.info('>' * 80)
            logger.info('%s' % datetime.datetime.now())

    def clip_eval(self, curr_iter, engine, logger):
        config = self.config
        engine.vos_model.eval()
        if is_main_process():
            logger.info('======== Start Evaluation ========')
        total_iou = []
        val_miou = 0.0
        if config['distributed']:
            with torch.no_grad():
                for i, sample in enumerate(self.val_loader):
                    image = sample['rgb'].cuda()
                    flow = sample['flow'].cuda()
                    mask = sample['gt'].cuda()
                    video_name = sample['info']['name'][0]
                    clip_length = config['clip_length']
                    clip_preds = engine.vos_model.module.clip_inference(clip_length, image, flow).cuda()
                    video_iou = db_eval_iou_multi(mask.cpu().detach().numpy(), clip_preds.cpu().detach().numpy())
                    if is_main_process():
                        logger.info(f"{video_name}: iou{video_iou}")
                    total_iou.append(video_iou)
            if is_main_process():
                val_miou = np.mean(total_iou)
                logger.info(f"iter: {curr_iter}, iou{val_miou}")
        else:
            with torch.no_grad():
                for i, sample in enumerate(self.val_loader):
                    image = sample['rgb'].cuda()
                    flow = sample['flow'].cuda()
                    mask = sample['gt'].cuda()
                    video_name = sample['info']['name'][0]
                    clip_length = config['clip_length']
                    clip_preds = engine.vos_model.clip_inference(clip_length, image, flow).cuda()
                    video_iou = db_eval_iou_multi(mask.cpu().detach().numpy(), clip_preds.cpu().detach().numpy())
                    logger.info(f"{video_name}: iou{video_iou}")
                    total_iou.append(video_iou)
            val_miou = np.mean(total_iou)
            logger.info(f"iter: {curr_iter}, iou{val_miou}")
        return val_miou
