# Loss functions

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import bbox_iou, bbox_alpha_iou, box_iou, box_giou, box_diou, box_ciou, xywh2xyxy
from utils.torch_utils import is_parallel




class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.



class SigmoidBin(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export






class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)



class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)



class RankSort(torch.autograd.Function):
    @staticmethod

    @staticmethod


class aLRPLoss(torch.autograd.Function):
    @staticmethod

    @staticmethod


class APLoss(torch.autograd.Function):
    @staticmethod

    @staticmethod


class ComputeLoss:
    # Compute losses




class ComputeLossOTA:
    # Compute losses





class ComputeLossBinOTA:
    # Compute losses





class ComputeLossAuxOTA:
    # Compute losses




