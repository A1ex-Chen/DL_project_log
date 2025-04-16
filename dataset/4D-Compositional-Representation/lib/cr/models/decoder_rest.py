import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.layers import (
    ResnetBlockFC, CResnetBlockConv1d,
    CBatchNorm1d, CBatchNorm1d_legacy,
)

class DecoderCBatchNorm(nn.Module):
    ''' Decoder class with CBN for ONet 4D.

    Args:
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned temporal code c
        dim (int): points dimension
        hidden_size (int): hidden dimension
        leaky (bool): whether to use leaky activation
        legacy (bool): whether to use legacy version
    '''


    # For ONet 4D
