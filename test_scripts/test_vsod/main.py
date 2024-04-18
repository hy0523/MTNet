import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import argparse
import os.path as osp
from evaluator import Eval_thread
from dataloader import EvalDataset


def main(cfg):
    if cfg.methods is None:
        method_names = os.listdir(cfg.pred_dir)
    else:
        method_names = cfg.methods.split(' ')
    if cfg.datasets is None:
        dataset_names = os.listdir(cfg.gt_dir)
    else:
        dataset_names = cfg.datasets.split(' ')

    threads = []
    for dataset in dataset_names:
        for method in method_names:
            loader = EvalDataset(img_root=osp.join(cfg.pred_dir, method, dataset),
                                 label_root=osp.join(cfg.gt_dir, dataset),
                                 use_flow=config.use_flow)
            thread = Eval_thread(loader, method, dataset)
            threads.append(thread)
    for thread in threads:
        print(thread.run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第一帧与最后一帧若--use_flow则不eval")
    parser.add_argument('--methods', type=str, default=None, help='字符串格式，算法名称')
    parser.add_argument('--datasets', type=str, default=None, help='验证的数据集，数据集之间用空格隔开')
    parser.add_argument('--gt_dir', type=str, default='./gt', help='文件名如果不是gt需要改dataloader.py中的文件名')
    parser.add_argument('--pred_dir', type=str, default='./result', help='文件名如果不是result需要改dataloader.py中的文件名')
    parser.add_argument('--use_flow', type=bool, default=True, help="如果使用光流则在第一帧和最后一帧GT上不eval【具体调整见dataloader】")

    config = parser.parse_args()
    main(config)
