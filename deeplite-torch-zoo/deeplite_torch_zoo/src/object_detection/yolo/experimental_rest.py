# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# This file contains experimental modules

import math

import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import get_activation
from deeplite_torch_zoo.src.dnn_blocks.ghostnetv2.ghostnet_blocks import GhostConv
from deeplite_torch_zoo.src.registries import VARIABLE_CHANNEL_BLOCKS
from deeplite_torch_zoo.utils import LOGGER


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(
                -torch.arange(1.0, n) / 2, requires_grad=True
            )  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.cat(y, 1)  # nms ensemble
        y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(
        self, c1, c2, k=1, s=1, p1=0, p2=0
    ):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))




# PicoDet blocks
# -------------------------------------------------------------------------


@VARIABLE_CHANNEL_BLOCKS.register()
class DWConvblock(nn.Module):
    "Depthwise conv + Pointwise conv"

    def __init__(self, in_channels, out_channels, k, s, act='relu'):
        super(DWConvblock, self).__init__()
        self.p = k // 2
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=k,
            stride=s,
            padding=self.p,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


@VARIABLE_CHANNEL_BLOCKS.register()
class CBH(nn.Module):
    def __init__(
        self, num_channels, num_filters, filter_size, stride, num_groups=1, act='hswish'
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            num_channels,
            num_filters,
            filter_size,
            stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(num_filters)
        self.act = get_activation(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))





class Ensemble(nn.ModuleList):
    # Ensemble of models



class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution


def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from deeplite_torch_zoo.src.object_detection.yolo.yolov5 import (
        Detect,
        DetectionModel,
    )

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        # ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
        ckpt = torch.load(w, map_location='cpu')  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(
            ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval()
        )  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (
            nn.Hardswish,
            nn.LeakyReLU,
            nn.ReLU,
            nn.ReLU6,
            nn.SiLU,
            Detect,
            DetectionModel,
        ):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    LOGGER.info(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[
        torch.argmax(torch.tensor([m.stride.max() for m in model])).int()
    ].stride  # max stride
    assert all(
        model[0].nc == m.nc for m in model
    ), f'Models have different class counts: {[m.nc for m in model]}'
    return model


# PicoDet blocks
# -------------------------------------------------------------------------


@VARIABLE_CHANNEL_BLOCKS.register()
class DWConvblock(nn.Module):
    "Depthwise conv + Pointwise conv"




@VARIABLE_CHANNEL_BLOCKS.register()
class CBH(nn.Module):




def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ES_SEModule(nn.Module):



@VARIABLE_CHANNEL_BLOCKS.register()
class ES_Bottleneck(nn.Module):

    @staticmethod

    @staticmethod



# YoloLite blocks
# -------------------------------------------------------------------------


class LC_SEModule(nn.Module):



@VARIABLE_CHANNEL_BLOCKS.register()
class LC_Block(nn.Module):



@VARIABLE_CHANNEL_BLOCKS.register()
class Dense(nn.Module):
        # self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        # self.fc = nn.Linear(num_filters, num_filters)



@VARIABLE_CHANNEL_BLOCKS.register()
class conv_bn_relu_maxpool(nn.Module):



@VARIABLE_CHANNEL_BLOCKS.register()
class Shuffle_Block(nn.Module):

    @staticmethod



class ADD(nn.Module):
    # Stortcut a list of tensors along dimension
