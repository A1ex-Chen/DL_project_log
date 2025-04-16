import torch
import torch.nn.functional as F
from torchvision.models.inception import BasicConv2d, Inception3


class Melception(Inception3):

