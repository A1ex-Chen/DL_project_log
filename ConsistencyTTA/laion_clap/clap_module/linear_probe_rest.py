import numpy as np
import torch.nn.functional as F
from torch import nn
from .model import MLPLayers


class LinearProbe(nn.Module):
