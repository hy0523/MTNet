from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import glob
from .imagecorruptions import corrupt
from .imagecorruptions.corruptions import *

corruption_tuple = (gaussian_noise, shot_noise, impulse_noise, defocus_blur,
                    glass_blur, motion_blur, zoom_blur, snow, frost, fog,
                    brightness, contrast, elastic_transform, pixelate,
                    jpeg_compression, speckle_noise, gaussian_blur, spatter,
                    saturate)

corruption_dict = {corr_func.__name__: corr_func for corr_func in
                   corruption_tuple}


class VideoTestDataset:
    def __init__(self, config,
                 img_dir=None,
                 flow_dir=None,
                 img_suffix='.jpg',
                 flow_suffix='.jpg',
                 drop_tail=True):
        self.config = config
        self.image_dir = img_dir
        self.flow_dir = flow_dir
        self.i_suffix = img_suffix
        self.f_suffix = flow_suffix
        self.drop_tail = drop_tail
        test_size = config['test_size']
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        resize = transforms.Resize(test_size)
        self.test_transform = transforms.Compose([resize, to_tensor, normalize])
        self.total_videos = sorted(os.listdir(self.image_dir))
        self.total_images = {}
        self.total_flows = {}
        total_frames = 0
        for video in self.total_videos:
            self.total_images[video] = sorted(glob.glob(os.path.join(self.image_dir, video, self.i_suffix)))
            total_frames += len(self.total_images[video])
            self.total_flows[video] = sorted(glob.glob(os.path.join(self.flow_dir, video, self.f_suffix)))
            if self.drop_tail:
                self.total_images[video] = self.total_images[video][:-1]
        self.total_frames = total_frames

    def __len__(self):
        return len(self.total_videos)

    def get_total_videos(self):
        return self.total_videos

    def get_total_frames(self):
        return self.total_frames

    def reader(self, img_path, flow_path, read_status='clear', severity=1):
        # first corruption, then resize
        img = self.rgb_loader(img_path)
        flow = self.rgb_loader(flow_path)
        if read_status == 'clear':
            pass
        else:
            img = np.asarray(img)  # numpy
            corrupted = corrupt(img, corruption_name=read_status, severity=severity)  # [1-5]
            img = Image.fromarray(corrupted)
        return img, flow

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class VOSTestDataset(Dataset):
    def __init__(self,
                 im_root='../../data/DAVIS2SEG/frame/val/',
                 fl_root='../../data/DAVIS2SEG/flow/val/',
                 test_size=(512, 512),
                 drop_tail=False,
                 skip_no_mask=False,
                 gt_root=None):
        self.im_root = im_root
        self.fl_root = fl_root
        self.main_images = []
        self.main_flows = []
        if skip_no_mask:
            print(f"Will skip no annotation images")
        for video_name in os.listdir(self.im_root):
            video_path = os.path.join(self.im_root, video_name)
            flow_path = os.path.join(self.fl_root, video_name)
            if skip_no_mask:
                maskfiles = sorted(glob.glob(os.path.join(gt_root, video_name, '*.jpg')) + glob.glob(
                    os.path.join(gt_root, video_name, '*.png')))
                accept_list = [os.path.basename(maskfile).split('.')[0] for maskfile in maskfiles]
                if drop_tail:
                    accept_list = accept_list[:-1]
                self.main_images += sorted([video_path + "/" + f for f in os.listdir(video_path) if
                                            os.path.basename(f).split('.')[0].split('_')[0] in accept_list])
                self.main_flows += sorted([flow_path + "/" + f for f in os.listdir(flow_path) if
                                           os.path.basename(f).split('.')[0].split('_')[0] in accept_list])
                assert len(self.main_images) == len(self.main_flows)
            else:
                if drop_tail:
                    self.main_images += sorted([video_path + "/" + f for f in os.listdir(video_path)])[:-1]
                else:
                    self.main_images += sorted([video_path + "/" + f for f in os.listdir(video_path)])
                self.main_flows += sorted([flow_path + "/" + f for f in os.listdir(flow_path)])
            assert len(self.main_images) == len(self.main_flows)
        print(f"Total {len(self.main_images)} test images")

        self.test_size = test_size

        self.img_flow_transform = transforms.Compose([
            transforms.Resize(self.test_size, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.gt_transform = transforms.Compose([
            transforms.Resize(self.test_size, interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.size = len(self.main_images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.main_images[index])
        flow = self.rgb_loader(self.main_flows[index])
        ori_size = image.size
        image = self.img_flow_transform(image)
        flow = self.img_flow_transform(flow)
        data = {'file_name': self.main_images[index],
                'rgb': image,
                'flow': flow,
                'ori_size': (ori_size)}
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
        return len(self.main_images)
