import os
import time
import torch
from torchvision import transforms
import importlib
from core.manager.engine import MTNetEngine
import math
import os.path as osp
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from IPython import embed
from core.dataloader.test_dataset import VideoTestDataset


class Evaluator(object):
    def __init__(self, config):
        self.config = config
        # build engine
        engine_config = importlib.import_module('core.networks.' + config['model_name'])
        model = engine_config.MTNet(config).to(config['device'])
        self.engine = MTNetEngine(model=model, config=config, optimizer=None)
        # prepare trained model
        self.process_trained_model()

    def prepare_test_transform(self):
        config = self.config
        test_size = config['test_size']
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        resize = transforms.Resize(test_size)
        self.image_transforms = transforms.Compose([resize, to_tensor, normalize])

    def clip_eval(self, task_name, test_dataset):
        self.prepare_test_transform()
        self.prepare_test_dataset(test_dataset)
        config = self.config
        result_dir = config['output_dir'] if config['output_dir'] else './output'
        project_name = config['model_name'] if config['model_name'] else 'MTNet'
        total_process_time = 0
        total_frames = 0
        test_length = config['test_length']
        model = self.engine.vos_model
        model_parameters = sum([p.data.nelement() for p in model.parameters()])
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        self.print_log("Number of model parameters: {}".format(model_parameters))
        self.print_log(("Model size {:.3f}MB".format(size_all_mb)))
        torch.cuda.set_device(device=0)
        model.cuda()
        model.eval()
        self.print_log(f'start evaluate {test_dataset}:')
        for video in tqdm(self.test_dataset.get_total_videos()):
            save_folder = '{}/{}/{}/{}/{}'.format(result_dir, project_name, task_name, test_dataset, video)
            os.makedirs(save_folder, exist_ok=True)

            imagefiles = self.test_dataset.total_images[video]
            flowfiles = self.test_dataset.total_flows[video]  # png

            total_frames += len(imagefiles)
            video_len = len(imagefiles)
            this_video_time = 0

            with torch.no_grad():
                if video_len > test_length:
                    num_clips = math.ceil(video_len / test_length)
                    for c in range(num_clips - 1):
                        start_idx = c * test_length
                        end_idx = (c + 1) * test_length
                        images = []
                        flows = []
                        for imagefile, flowfile in zip(imagefiles[start_idx:end_idx], flowfiles[
                                                                                      start_idx:end_idx]):  # todo: solve memory overflow problem

                            image, flow = self.test_dataset.reader(img_path=imagefile, flow_path=flowfile,
                                                                   read_status='clear', severity=0)
                            width, height = image.size
                            image = self.test_dataset.test_transform(image)
                            flow = self.test_dataset.test_transform(flow)
                            image = image.unsqueeze(0)
                            flow = flow.unsqueeze(0)
                            image, flow = image.cuda(), flow.cuda()
                            images.append(image)
                            flows.append(flow)
                        clip_images = torch.stack(images, dim=1)
                        clip_flows = torch.stack(flows, dim=1)
                        start_time = time.time()
                        clip_output = model(clip_images, clip_flows)
                        this_video_time += time.time() - start_time
                        clip_output = clip_output.cpu().detach()
                        for i, imagefile in enumerate(imagefiles[start_idx:end_idx]):
                            this_mask_pred = clip_output[i:i + 1, 0:1, :, :]  # tensor
                            this_mask_pred = F.interpolate(this_mask_pred, size=((height, width)), mode='bilinear',
                                                           align_corners=True)

                            this_mask_pred = (this_mask_pred - this_mask_pred.min()) / (
                                    this_mask_pred.max() - this_mask_pred.min() + 1e-8)
                            if task_name == 'UVOS':
                                this_mask_pred[this_mask_pred > 0.5] = 1
                                this_mask_pred[this_mask_pred <= 0.5] = 0
                            this_mask_pred = Image.fromarray(this_mask_pred[0, 0].cpu().detach().numpy() * 255).convert(
                                'L')
                            save_file = os.path.join(save_folder, os.path.basename(imagefile)[:-4] + '.png')
                            this_mask_pred.save(save_file)
                    if video_len > end_idx:
                        res_images = []
                        res_flows = []
                        for imagefile, flowfile in zip(imagefiles[end_idx:],
                                                       flowfiles[end_idx:]):  # todo: solve the memory overflow problem
                            image, flow = self.test_dataset.reader(img_path=imagefile, flow_path=flowfile,
                                                                   read_status='clear', severity=0)
                            width, height = image.size
                            image = self.test_dataset.test_transform(image)
                            flow = self.test_dataset.test_transform(flow)
                            image = image.unsqueeze(0)
                            flow = flow.unsqueeze(0)
                            image, flow = image.cuda(), flow.cuda()
                            res_images.append(image)
                            res_flows.append(flow)
                        if res_images == []:
                            embed()
                        res_images = torch.stack(res_images, dim=1)
                        res_flows = torch.stack(res_flows, dim=1)
                        start_time = time.time()
                        res_output = model(res_images, res_flows).cpu().detach()
                        this_video_time += time.time() - start_time
                        for i, imagefile in enumerate(imagefiles[end_idx:]):
                            this_mask_pred = res_output[i:i + 1, 0:1, :, :]  # tensor
                            this_mask_pred = F.interpolate(this_mask_pred, size=((height, width)), mode='bilinear',
                                                           align_corners=True)
                            this_mask_pred = (this_mask_pred - this_mask_pred.min()) / (
                                    this_mask_pred.max() - this_mask_pred.min() + 1e-8)
                            if task_name == 'UVOS':
                                this_mask_pred[this_mask_pred > 0.5] = 1
                                this_mask_pred[this_mask_pred <= 0.5] = 0
                            this_mask_pred = Image.fromarray(this_mask_pred[0, 0].cpu().detach().numpy() * 255).convert(
                                'L')
                            save_file = os.path.join(save_folder, os.path.basename(imagefile)[:-4] + '.png')
                            this_mask_pred.save(save_file)
                else:
                    short_images = []
                    short_flows = []
                    for imagefile, flowfile in zip(imagefiles, flowfiles):  # todo: solve memory overflow problem
                        image = Image.open(imagefile).convert('RGB')
                        flow = Image.open(flowfile).convert('RGB')
                        width, height = image.size
                        image = self.image_transforms(image)
                        flow = self.image_transforms(flow)
                        image = image.unsqueeze(0)
                        flow = flow.unsqueeze(0)
                        image, flow = image.cuda(), flow.cuda()
                        short_images.append(image)
                        short_flows.append(flow)
                    short_images = torch.stack(short_images, dim=1)
                    short_flows = torch.stack(short_flows, dim=1)
                    start_time = time.time()
                    short_output = model(short_images, short_flows)
                    this_video_time += time.time() - start_time
                    short_output = short_output.cpu().detach()
                    for i, imagefile in enumerate(imagefiles):
                        this_mask_pred = short_output[i:i + 1, 0:1, :, :]  # tensor
                        this_mask_pred = F.interpolate(this_mask_pred, size=((height, width)), mode='bilinear',
                                                       align_corners=True)
                        this_mask_pred = (this_mask_pred - this_mask_pred.min()) / (
                                this_mask_pred.max() - this_mask_pred.min() + 1e-8)
                        if task_name == 'UVOS':
                            this_mask_pred[this_mask_pred > 0.5] = 1
                            this_mask_pred[this_mask_pred <= 0.5] = 0
                        this_mask_pred = Image.fromarray(this_mask_pred[0, 0].cpu().detach().numpy() * 255).convert(
                            'L')
                        save_file = os.path.join(save_folder, os.path.basename(imagefile)[:-4] + '.png')
                        this_mask_pred.save(save_file)
                total_process_time += this_video_time

        self.print_log('done!')
        self.print_log(f"Total processing time: {total_process_time}")
        self.print_log(f"Total processed frames: {total_frames}")
        self.print_log(f"FPS: {total_frames / total_process_time}\n")

    def prepare_test_dataset(self, test_dataset):
        config = self.config
        dataset_root = config['root']  # './data'
        self.img_suffix = '*.jpg'
        self.flow_suffix = '*.png'
        self.drop_tail = False
        if test_dataset == 'LongVideos':
            self.drop_tail = True
            self.flow_suffix = '*.png'
            self.image_dir = osp.join(dataset_root, test_dataset, 'JPEGImages')
            self.flow_dir = osp.join(dataset_root, test_dataset, 'Flow')

        if test_dataset == 'DAVIS16':
            self.flow_suffix = '*.jpg'
            self.image_dir = osp.join(dataset_root, test_dataset, 'val', 'frame')
            self.flow_dir = osp.join(dataset_root, test_dataset, 'val', 'flow')

        if test_dataset == 'YO2SEG':
            self.flow_suffix = '*.jpg'
            self.image_dir = osp.join(dataset_root, test_dataset, 'val', 'frame')
            self.flow_dir = osp.join(dataset_root, test_dataset, 'val', 'flow')

        if test_dataset == 'FBMS-59':
            self.drop_tail = True
            self.flow_rex = '*.png'
            self.image_dir = osp.join(dataset_root, test_dataset, 'test', 'frame')
            self.flow_dir = osp.join(dataset_root, test_dataset, 'test', 'flow')

        if test_dataset == 'SegTrack-V2':
            self.drop_tail = True
            self.flow_suffix = '*.png'
            self.image_dir = osp.join(dataset_root, test_dataset, 'test', 'frame')
            self.flow_dir = osp.join(dataset_root, test_dataset, 'test', 'flow')

        if test_dataset == 'Easy-35':
            self.drop_tail = True
            self.img_suffix = '*.png'
            self.flow_suffix = '*.png'
            self.image_dir = osp.join(dataset_root, test_dataset, 'test', 'frame')
            self.flow_dir = osp.join(dataset_root, test_dataset, 'test', 'flow')

        if test_dataset == 'MCL':
            self.drop_tail = True
            self.flow_suffix = '*.png'
            self.image_dir = osp.join(dataset_root, test_dataset, 'test', 'frame')
            self.flow_dir = osp.join(dataset_root, test_dataset, 'test', 'flow')

        if test_dataset == 'ViSal':
            self.drop_tail = True
            self.flow_suffix = '*.png'
            self.image_dir = osp.join(dataset_root, test_dataset, 'test', 'frame')
            self.flow_dir = osp.join(dataset_root, test_dataset, 'test', 'flow')

        self.test_dataset = VideoTestDataset(config=config,
                                             img_dir=self.image_dir,
                                             flow_dir=self.flow_dir,
                                             img_suffix=self.img_suffix,
                                             flow_suffix=self.flow_suffix,
                                             drop_tail=self.drop_tail)

    def process_trained_model(self):
        config = self.config
        test_model = config['test_model']
        self.engine.load_network(test_model)

    def print_log(self, msg):
        print(msg)
