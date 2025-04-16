import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.assigners.iou2d_calculator import iou2d_calculator
from yolov6.assigners.assigner_utils import dist_calculator, select_candidates_in_gts, select_highest_overlaps, iou_calculator

class ATSSAssigner(nn.Module):
    '''Adaptive Training Sample Selection Assigner'''

    @torch.no_grad()


