import torch
import torch.nn.functional as F
from torchvision.models.inception import BasicConv2d, Inception3
from collections import OrderedDict




class Melception(Inception3):

