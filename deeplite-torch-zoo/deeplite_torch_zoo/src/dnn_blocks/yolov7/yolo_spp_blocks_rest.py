# Taken from:
# https://github.com/WongKinYiu/yolov7/blob/HEAD/models/common.py

import warnings

import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct, get_activation
from deeplite_torch_zoo.src.registries import VARIABLE_CHANNEL_BLOCKS


@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOSPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP



@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOSPPCSP(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks



@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOSPPCSPLeaky(nn.Module):



@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOSPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher



@VARIABLE_CHANNEL_BLOCKS.register()
class YOLOSPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
