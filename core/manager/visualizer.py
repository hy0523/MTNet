import os
import numpy as np
from tqdm import tqdm
import cv2


class Visualizer:
    def __init__(self, config, model=None):
        self.config = config
        self.model = model
        self.colors = {'blue': (255, 0, 0), 'red': (0, 0, 255), 'green': (0, 255, 0)}
        for key, value in self.colors.items():
            self.colors[key] = tuple([c / 255 for c in self.colors[key]])
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.vis_path = './vis'
        os.makedirs(self.vis_path, exist_ok=True)

    def dataset_apply_mask(self):
        config = self.config
        dataset_names = config['dataset_name']
        img_root = config['img_root']
        pred_root = config['pred_root']
        # mask_root = config['mask_root']
        save_root = config['save_root'] if config['save_root'] else self.vis_path
        os.makedirs(save_root, exist_ok=True)
        for _, dataset_name in enumerate(dataset_names):
            if dataset_name == 'DAVIS16' or dataset_name == 'YO2SEG':
                dataset_img_root = os.path.join(img_root, dataset_name, 'val', 'frame')
            elif dataset_name == 'LongVideos':
                dataset_img_root = os.path.join(img_root, dataset_name, 'JPEGImages')
            elif dataset_name == 'FBMS-59':
                dataset_img_root = os.path.join(img_root, dataset_name, 'test', 'frame')
            elif dataset_name == 'ViSal':
                dataset_img_root = os.path.join(img_root, dataset_name, 'test', 'frame')
            elif dataset_name == 'SegTrack-V2':
                dataset_img_root = os.path.join(img_root, dataset_name, 'test', 'frame')
            elif dataset_name == 'MCL':
                dataset_img_root = os.path.join(img_root, dataset_name, 'test', 'frame')
            elif dataset_name == 'Easy-35':
                dataset_img_root = os.path.join(img_root, dataset_name, 'test', 'frame')
            else:
                dataset_img_root = None
            print(f"Processing {dataset_name} datasets... ")
            dataset_pred_root = os.path.join(pred_root, dataset_name)
            dataset_save_root = os.path.join(save_root, config['model_name'], dataset_name)
            videos = sorted(os.listdir(dataset_img_root))
            for _, video in enumerate(tqdm(videos)):
                video_img_path = os.path.join(dataset_img_root, video)
                video_pred_path = os.path.join(dataset_pred_root, video)
                video_save_path = os.path.join(dataset_save_root, video)
                os.makedirs(video_save_path, exist_ok=True)
                imgs_path = [os.path.join(video_img_path, img_name) for img_name in os.listdir(video_img_path)]
                preds_path = [os.path.join(video_pred_path, img_name) for img_name in os.listdir(video_pred_path)]
                imgs_path = sorted(imgs_path)
                preds_path = sorted(preds_path)
                if len(imgs_path) > len(preds_path):
                    imgs_path = imgs_path[:-1]
                assert len(imgs_path) == len(preds_path)
                for _, (img_path, pred_path) in enumerate(zip(imgs_path, preds_path)):
                    img_name = os.path.basename(img_path)
                    save_path = os.path.join(video_save_path, img_name)
                    img = cv2.imread(img_path)
                    pred = cv2.imread(pred_path, 0)  # [0,255]
                    pred[pred == 255] = 1
                    masked_img = self.apply_mask(image=img, mask=pred, color=self.colors['red'], alpha=0.5)
                    # masked_img = self.mask_cover(img=img, mask=pred, scale=0.5)
                    cv2.imwrite(save_path, masked_img)
                if config['make_video']:
                    from core.utils.video_tool import VideoMaker
                    img = cv2.imread(imgs_path[0])
                    img_size = (img.shape[1], img.shape[0])
                    video_name = video_save_path.split('/')[-1]
                    video_dir = os.path.join(video_save_path, video_name + '.mp4')  # avi mp4
                    video_maker = VideoMaker(frames_dir=video_save_path,
                                             out_dir=video_dir,
                                             fps=20,
                                             img_size=img_size)
                    video_maker.make_video()
                    video_maker.release()

    def apply_mask(self, image, mask, color, alpha=0.5):
        # mask = F
        r""" Apply mask to the given image. """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    def to_color(self, seg, PALETTE):
        h, w = seg.shape
        color_seg = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in enumerate(PALETTE):
            color_seg[seg == label, :] = color
        return color_seg

    def mask_cover(self, img, mask, scale=0.5):
        # PALETTE = [[0, 0, 0], [50, 50, 255]]
        PALETTE = [[0, 0, 0], [0, 0, 255]]
        color_seg = self.to_color(mask.copy(), PALETTE)
        mask = img * (1 - scale) + color_seg * scale
        return mask
