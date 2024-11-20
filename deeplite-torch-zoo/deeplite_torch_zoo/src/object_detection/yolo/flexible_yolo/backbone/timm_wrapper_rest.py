import timm
import torch.nn as nn

from deeplite_torch_zoo.utils import LOGGER


class TimmWrapperBackbone(nn.Module):
    """
    Wrapper to use backbones from timm
    https://github.com/huggingface/pytorch-image-models
    """

