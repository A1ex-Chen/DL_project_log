import torch
import numpy as np
from torch.nn import functional as F
from lib.common import compute_iou
from lib.training import BaseTrainer


class Trainer(BaseTrainer):
    r''' Trainer object for ONet 4D.

    Onet 4D is trained with BCE. The Trainer object
    obtains methods to perform a train and eval step as well as to visualize
    the current training state.

    Args:
        model (nn.Module): Onet 4D model
        optimizer (PyTorch optimizer): The optimizer that should be used
        device (PyTorch device): the PyTorch device
        input_type (string): The input type (e.g. 'img')
        vis_dir (string): the visualisation directory
        threshold (float): threshold value for decision boundary
    '''












