import torch
import torch.nn as nn
import torch.nn.functional as F
from .basiclayers.basiclayers import VGGBlock


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        vgg = [96, 96, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        out_channels = 10
        self.model = VGGBlock(structure=vgg, in_channels=3, out_channels=out_channels)

    def forward(self, x):
        out = self.model(x)
        return out








