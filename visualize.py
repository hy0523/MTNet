import sys

sys.path.append('.')
sys.path.append('..')
from core.utils.args import Configuration
from core.manager.visualizer import Visualizer
import os

config = Configuration()
config.get_parser()
config['dataset_name'] = ['DAVIS16']  # ['DAVIS16', 'FBMS-59', 'LongVideos', 'YO2SEG', 'MCL', 'ViSal', 'SegTrack-V2', 'Easy-35']
config['img_root'] = '../data'
config['pred_root'] = os.path.join(config['output_dir'], config['model_name'], config['task_name'])
config['save_root'] = 'qualitative'
config['make_video'] = True
visualizer = Visualizer(config)
visualizer.dataset_apply_mask()
