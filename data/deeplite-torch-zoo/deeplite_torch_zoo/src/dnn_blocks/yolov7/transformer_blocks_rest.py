# Taken from:
# https://github.com/WongKinYiu/yolov7/blob/HEAD/models/common.py
# The file is modified by Deeplite Inc. from the original implementation on Nov 29, 2022
# Refactoring (subset from the aforemnetioned source file)

import torch
from torch import nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct
from deeplite_torch_zoo.src.dnn_blocks.yolov7.transformer_common import (
    SwinTransformerLayer,
    SwinTransformerLayer_v2,
    TransformerLayer,
)


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929



class SwinTransformerBlock(nn.Module):



class SwinTransformer2Block(nn.Module):



class STCSPA(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks



class STCSPB(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks



class STCSPC(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
