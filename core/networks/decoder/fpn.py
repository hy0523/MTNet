import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed


class FPN(nn.Module):
    def __init__(self,
                 in_channels: list = [256, 512, 1024, 2048]):
        super(FPN, self).__init__()
        # Top layer
        self.toplayer = nn.Conv2d(in_channels[-1], in_channels[0], kernel_size=1, stride=1,
                                  padding=0)  # Reduce channels

        # Smooth layer
        self.smooth1 = nn.Sequential(nn.Conv2d(in_channels[0], in_channels[0], kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(in_channels[0]),
                                     nn.ReLU())
        self.smooth2 = nn.Sequential(nn.Conv2d(in_channels[0], in_channels[0], kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(in_channels[0]),
                                     nn.ReLU())
        self.smooth3 = nn.Sequential(nn.Conv2d(in_channels[0], in_channels[0], kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(in_channels[0]),
                                     nn.ReLU())

        # Lateral layers
        self.latlayer1 = nn.Sequential(nn.Conv2d(in_channels[-2], in_channels[0], kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(in_channels[0]),
                                       nn.ReLU())
        self.latlayer2 = nn.Sequential(nn.Conv2d(in_channels[-3], in_channels[0], kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(in_channels[0]),
                                       nn.ReLU())
        self.latlayer3 = nn.Sequential(nn.Conv2d(in_channels[-4], in_channels[0], kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(in_channels[0]),
                                       nn.ReLU())

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, c2, c3, c4, c5):
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5


