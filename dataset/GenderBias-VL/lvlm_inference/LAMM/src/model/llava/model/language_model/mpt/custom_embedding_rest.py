import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SharedEmbedding(nn.Embedding):
