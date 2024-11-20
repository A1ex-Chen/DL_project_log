import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.assigners.assigner_utils import select_candidates_in_gts, select_highest_overlaps, iou_calculator, dist_calculator

class TaskAlignedAssigner(nn.Module):

    @torch.no_grad()



