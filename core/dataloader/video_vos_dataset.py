import os
from os import path as osp
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
from IPython import embed
import cv2
from core.dataloader.torchvideotransforms import video_transforms, volume_transforms
import random
from PIL import ImageEnhance
import matplotlib.pyplot as plt


def cv_random_flip(images, labels, flows):
    flip_flag = random.randint(0, 1)

    if flip_flag == 1:
        images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
        labels = [label.transpose(Image.FLIP_LEFT_RIGHT) for label in labels]
        flows = [flow.transpose(Image.FLIP_LEFT_RIGHT) for flow in flows]

    return images, labels, flows


def randomCrop(images, labels, flows):
    border = 30
    image_width = images[0].size[0]
    image_height = images[0].size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)

    images = [image.crop(random_region) for image in images]
    labels = [label.crop(random_region) for label in labels]
    flows = [flow.crop(random_region) for flow in flows]
    return images, labels, flows


def randomRotation(images, labels, flows):
    mode = Image.BICUBIC

    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        images = [image.rotate(random_angle, mode) for image in images]
        labels = [label.rotate(random_angle, mode) for label in labels]
        flows = [flow.rotate(random_angle, mode) for flow in flows]
        # image = image.rotate(random_angle, mode)
        # label = label.rotate(random_angle, mode)
        # flow = flow.rotate(random_angle, mode)

    return images, labels, flows


def colorEnhance(images):
    # No hue change here as that's not realistic
    bright_intensity = random.randint(5, 15) / 10.0
    images = [ImageEnhance.Brightness(image).enhance(bright_intensity) for image in images]

    contrast_intensity = random.randint(5, 15) / 10.0
    images = [ImageEnhance.Contrast(image).enhance(contrast_intensity) for image in images]

    color_intensity = random.randint(0, 20) / 10.0
    images = [ImageEnhance.Color(image).enhance(color_intensity) for image in images]

    sharp_intensity = random.randint(0, 30) / 10.0
    images = [ImageEnhance.Sharpness(image).enhance(sharp_intensity) for image in images]

    return images


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)

        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])

    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])

    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255

    return Image.fromarray(img)


class VOSDataset(Dataset):
    def __init__(self,
                 im_root: str,
                 fl_root: str,
                 gt_root: str,
                 max_jump: int = 3,
                 subset: str = None,
                 inputRes=(512, 512),
                 num_frames=3):
        self.im_root = im_root
        self.fl_root = fl_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.videos = []
        self.frames = {}
        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < num_frames:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)
        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))
        self.train_size = inputRes
        video_target_transform_list = [
            video_transforms.Resize(size=(self.train_size)),
        ]
        self.all_augment_transform_clip = video_transforms.Compose(video_target_transform_list)
        video_only_transform_list = [
            volume_transforms.ClipToTensor(channel_nb=3),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.video_augment_transform = video_transforms.Compose(video_only_transform_list)
        target_only_transform_list = [
            volume_transforms.ClipToTensor(channel_nb=1),
        ]
        self.target_augment_transform = video_transforms.Compose(target_only_transform_list)

    def __getitem__(self, id):
        video = self.videos[id]
        info = {}
        info['name'] = video
        info['frames'] = []
        vid_im_path = osp.join(self.im_root, video)
        vid_fl_path = osp.join(self.fl_root, video)
        vid_gt_path = osp.join(self.gt_root, video)
        frames = self.frames[video]
        this_max_jump = min(len(frames), self.max_jump)
        start_idx = np.random.randint(len(frames) - this_max_jump + 1)
        f1_idx = start_idx + np.random.randint(this_max_jump + 1) + 1
        f1_idx = min(f1_idx, len(frames) - this_max_jump, len(frames) - 1)
        f2_idx = f1_idx + np.random.randint(this_max_jump + 1) + 1
        f2_idx = min(f2_idx, len(frames) - this_max_jump // 2, len(frames) - 1)
        frames_idx = [start_idx, f1_idx, f2_idx]
        if np.random.rand() < 0.5:
            frames_idx = frames_idx[::-1]
        # print(frames_idx)
        images = []
        flows = []
        masks = []
        for f_idx in frames_idx:
            jpg_name = frames[f_idx][:-4] + '.jpg'
            png_name = frames[f_idx][:-4] + '.png'
            info['frames'].append(jpg_name)
            this_im = self.rgb_loader(os.path.join(vid_im_path, jpg_name))
            this_fl = self.rgb_loader(os.path.join(vid_fl_path, jpg_name))
            this_gt = cv2.imread(os.path.join(vid_gt_path, png_name), 0)
            this_gt[this_gt > 0] = 255
            this_gt = Image.fromarray(this_gt)
            images.append(this_im)
            flows.append(this_fl)
            masks.append(this_gt)
        images, masks, flows = cv_random_flip(images, masks, flows)
        images, masks, flows = randomCrop(images, masks, flows)
        images, masks, flows = randomRotation(images, masks, flows)

        images = colorEnhance(images)
        images, flows, masks = self.all_augment_transform_clip(images, flows, masks)
        images = self.video_augment_transform(images).transpose(0, 1)
        flows = self.video_augment_transform(flows).transpose(0, 1)
        masks = self.target_augment_transform(masks).transpose(0, 1)
        data = {
            'rgb': images,
            'flow': flows,
            'gt': masks,
            'info': info,
        }
        return data

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return len(self.videos)
