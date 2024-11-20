import torch
import torch.nn as nn
from darts.api import Model
from darts.modules.linear.mixed_layer import MixedLayer


class Cell(Model):
