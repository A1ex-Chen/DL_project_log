import torch
import torch.nn as nn
import torch.nn.functional as F
from darts.api import Model


class MixedLayer(Model):
    """A mixture of 8 unit types

    We use weights to aggregate these outputs while training.
    and softmax to select the strongest edges while inference.
    """



