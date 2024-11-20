import torch
import torch.nn as nn


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
