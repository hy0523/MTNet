import sys

sys.path.append('.')
sys.path.append('..')
from core.utils.args import Configuration
from core.manager.evaluator import Evaluator
import torch

# 'UVOS':['DAVIS16', 'YO2SEG', 'FBMS-59', 'LongVideos']
# 'VSOD':['DAVIS16','Easy-35','FBMS-59', 'MCL', 'ViSal']

config = Configuration()
config.get_parser()
evaluator = Evaluator(config)
with torch.cuda.amp.autocast(enabled=not config['benchmark']):
    evaluator.clip_eval(task_name=config['task_name'], test_dataset=config['test_dataset'])
