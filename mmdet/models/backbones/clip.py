import warnings
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.registry import MODELS
import torch
import clip
from PIL import Image
import numpy as np
import torch.nn.functional as F
import cv2

###### using clip for backbone
@MODELS.register_module()
class Clip(BaseModule):
    def __init__(self,
                 type='ViT-B/32',):
        super(Clip, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip, self.preprocess = clip.load(type, device=self.device)
    def forward(self, x):
        frame = x.size(1) #torch.Size([3, 5, 3, 512, 512])
        features = []
        for i in range(frame):
            x_i = x[:,i,:,:,:]
            x_i = self.preprocess(x_i)
            # with torch.no_grad():
            fea = self.clip.encode_image(x_i)
            fea = F.interpolate(fea.unsqueeze(0), size=(x.size(3), x.size(4)), mode='bilinear', align_corners=False)
            features.append(fea)
        return features