# Ultralytics YOLO 🚀, AGPL-3.0 license

import math

import torch
import torch.nn as nn

from deeplite_torch_zoo.src.dnn_blocks.common import ConvBnAct as Conv
from deeplite_torch_zoo.utils import check_version


class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(
            b, 4, a
        )
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)





        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = (
            torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        )  # shift x
        sy = (
            torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        )  # shift y
        sy, sx = (
            torch.meshgrid(sy, sx, indexing='ij')
            if check_version(torch.__version__, '1.10.0')
            else torch.meshgrid(sy, sx)
        )
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


class DetectV8(nn.Module):
    # YOLOv8 Detect head for detection models
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    no_post_processing = False  # don't export bbox decoding ops


