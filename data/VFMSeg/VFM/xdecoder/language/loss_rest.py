import pickle
from distutils import log

import torch
import torch.nn.functional as F
import torch.distributed as dist

from einops import rearrange, repeat
from timm.loss import SoftTargetCrossEntropy

soft_cross_entropy = SoftTargetCrossEntropy()












