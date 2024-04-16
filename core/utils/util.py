import os
import numpy as np
from PIL import Image
import random
import logging

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.init as initer


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9, index_split=-1, scale_lr=10., warmup=False,
                       warmup_step=500):
    """poly learning rate policy"""
    if warmup and curr_iter < warmup_step:
        lr = base_lr * (0.1 + 0.9 * (curr_iter / warmup_step))
    else:
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** power

    optimizer.param_groups[0]["lr"] = lr * 0.1
    optimizer.param_groups[1]["lr"] = lr


def setup_seed(seed=2022, deterministic=False):
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
