"""
modules.py - This file stores the rathering boring network blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model import mod_resnet
from model import cbam


class ResBlock(nn.Module):
    """
    Explanation: ResNet like block with some modifications, padding 1 with kernel size 3 will not change the size of the feature map.
    Concatenate the last layer features from both key and query encoder -> avoid bloating feature dimensions.
    """
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class FeatureFusionBlock(nn.Module):
    """
    Explaination: This block is used to fuse the feature from input image and f16 feature from key encoder.
    """
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.attention = cbam.CBAM(outdim)
        self.block2 = ResBlock(outdim, outdim)

    def forward(self, x, f16):
        """
        f16 is feature from key encoder. 
        TODO: review later.
        """
        x = torch.cat([x, f16], dim=1)
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)

        return x


# Single object version, used only in static image pretraining
# This will be loaded and modified into the multiple objects version later (in stage 1/2/3)
# See model.py (load_network) for the modification procedure
class ValueEncoderSO(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet18(pretrained=True, extra_chan=1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 64
        self.layer2 = resnet.layer2 # 1/8, 128
        self.layer3 = resnet.layer3 # 1/16, 256

        self.fuser = FeatureFusionBlock(1024 + 256, 512)

    def forward(self, image, key_f16, mask):
        # key_f16 is the feature from the key encoder

        f = torch.cat([image, mask], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 64
        x = self.layer2(x) # 1/8, 128
        x = self.layer3(x) # 1/16, 256

        x = self.fuser(x, key_f16)

        return x


# Multiple objects version, used in other times
class ValueEncoder(nn.Module):
    """
    Explanation: image and mask as inputs, encode with ResNet18
    """
    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet18(pretrained=True, extra_chan=2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu     # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 64
        self.layer2 = resnet.layer2 # 1/8, 128
        self.layer3 = resnet.layer3 # 1/16, 256

        self.fuser = FeatureFusionBlock(1024 + 256, 512)

    def forward(self, image, key_f16, mask, other_masks):
        """
        Training: Encode ONE image (T=1), kf16 feature from key encoder, mask of one object, and other masks of second object
            image: (N, 3, H, W)
            key_f16: (N, 1024, H/16, W/16)
            mask: (N, 1, H, W)
            other_masks: (N, 1, H, W)
        Inference: TODO
        """
        # key_f16 is the feature from the key encoder

        f = torch.cat([image, mask, other_masks], 1) # (N, 5, H, W)
        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)    # (N, 64, H/2, W/2)
        x = self.maxpool(x) # (N, 64, H/4, W/4)
        x = self.layer1(x)  # (N, 64, H/4, W/4)
        x = self.layer2(x)  # (N, 128, H/8, W/8)
        x = self.layer3(x)  # (N, 256, H/16, W/16)

        x = self.fuser(x, key_f16) # (B, 256 + 256, H/16, W/16)

        return x


class KeyEncoder(nn.Module):
    """
    Explanation: encode query image with ResNet50, take only image as input.
    """
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024
        # Skip resnet.layer4, since we don't need the feature from 1/32

    def forward(self, img):
        """
        Input: img (B*T, C=3, H, W)
        """
        x = self.conv1(img) 
        x = self.bn1(x)
        x = self.relu(x)        # (B*T, 64, H/2, W/2)
        x = self.maxpool(x)     # (B*T, 64, H/4, W/4)
        f4 = self.res2(x)       # (B*T, 256, H/4, W/4) 
        f8 = self.layer2(f4)    # (B*T, 512, H/8, W/8)
        f16 = self.layer3(f8)   # (B*T, 1024, H/16, W/16)

        return f16, f8, f4


class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class KeyProjection(nn.Module):
    """
    Explanation: project the feature to the same number of channels as the key space.
    """
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x):
        return self.key_proj(x)
