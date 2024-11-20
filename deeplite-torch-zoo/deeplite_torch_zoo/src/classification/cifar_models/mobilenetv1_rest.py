# Code taken from: https://github.com/kuangliu/pytorch-cifar

import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Depthwise conv + Pointwise conv"""




class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [
        64,
        (128, 2),
        128,
        (256, 2),
        256,
        (512, 2),
        512,
        512,
        512,
        512,
        512,
        (1024, 2),
        1024,
    ]


