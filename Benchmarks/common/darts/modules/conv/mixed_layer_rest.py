import torch
import torch.nn as nn
from darts.api import Model
from darts.genotypes import PRIMITIVES
from darts.modules.operations.conv import OPS


class MixedLayer(Model):
    """A mixture of 8 unit types

    We use weights to aggregate these outputs while training.
    and softmax to select the strongest edges while inference.
    """


