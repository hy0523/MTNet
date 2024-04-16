"""
Modified from https://github.com/seoungwugoh/STM/blob/master/dataset.py
"""
import sys

sys.path.append('.')
sys.path.append('..')
import os

from os import path as osp
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data.dataset import Dataset
from core.dataloader.range_transform import im_normalization



class DAVISTestDataset(Dataset):
    def __init__(self,
                 im_root,
                 fl_root,
                 gt_root,
                 imset='2016/val.txt',
                 target_name=None,
                 size=(512, 512)):
        self.im_root = im_root
        self.fl_root = fl_root
        self.gt_root = gt_root
        self.mask_dir = osp.join(gt_root)
        self.image_dir = osp.join(im_root)
        self.flow_dir = osp.join(fl_root)
        _imset_dir = osp.join(self.im_root, 'ImageSets')
        _imset_f = osp.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        lines = os.listdir(self.image_dir)
        for video in lines:
            if target_name is not None and target_name != video:
                continue
            self.videos.append(video)
            self.num_frames[video] = len(os.listdir(osp.join(self.image_dir, video)))
            mask_plattle = sorted(os.listdir(osp.join(self.mask_dir, video)))[0]
            _mask = np.array(Image.open(osp.join(self.mask_dir, video, mask_plattle)).convert("P"))
            self.num_objects[video] = np.max(_mask)
            self.shape[video] = np.shape(_mask)

        self.size = size
        self.im_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            im_normalization])

        self.mask_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.videos)

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)

            return img.convert('L')

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['frames'] = []
        info['flows'] = []
        info['num_frames'] = self.num_frames[video]

        images = []
        flows = []
        masks = []
        for f in range(self.num_frames[video]):
            img_file = osp.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            flow_file = osp.join(self.flow_dir, video, '{:05d}.jpg'.format(f))  # jpg,png...
            images.append(self.im_transform(Image.open(img_file).convert('RGB')))
            flows.append(self.im_transform(Image.open(flow_file).convert('RGB')))
            info['frames'].append('{:05d}.jpg'.format(f))
            info['flows'].append('{:05d}.jpg'.format(f))
            mask_file = osp.join(self.mask_dir, video, '{:05d}.png'.format(f))
            if osp.exists(mask_file):
                mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
                mask = mask[np.newaxis, :, :]
                masks.append(mask)
            else:
                # Test-set maybe?
                print("Mask is none, please check out here!")
                masks.append(np.zeros_like(masks[0]))
        images = torch.stack(images, 0)
        flows = torch.stack(flows, 0)
        # masks = torch.stack(masks, 0)
        masks = np.stack(masks, 0)
        masks = (masks > 0.5).astype(np.uint8)
        masks = torch.from_numpy(masks).float()
        masks = self.mask_transform(masks)  # masks.unique()
        data = {
            'rgb': images,
            'flow': flows,
            'gt': masks,
            'info': info
        }
        return data
