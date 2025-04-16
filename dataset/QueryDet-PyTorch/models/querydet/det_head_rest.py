# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
import numpy as np
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn
import torch.nn.functional as F 

from detectron2.layers import ShapeSpec, batched_nms, cat, Conv2d, get_norm
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from detectron2.modeling.roi_heads.roi_heads import ROIHeads
from detectron2.modeling.poolers import ROIPooler


class RetinaNetHead_3x3(nn.Module):

    
        

class Head_3x3(nn.Module):




from utils.merged_sync_bn import MergedSyncBatchNorm

class RetinaNetHead_3x3_MergeBN(nn.Module):



    





class Head_3x3_MergeBN(nn.Module):

    
    
    

