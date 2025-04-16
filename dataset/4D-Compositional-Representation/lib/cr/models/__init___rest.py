import torch
import torch.nn as nn
from torch import distributions as dist
from lib.cr.models import decoder, velocity_field
from lib.utils.torchdiffeq.torchdiffeq import odeint, odeint_adjoint



decoder_dict = {
    'cbatchnorm': decoder.DecoderCBatchNorm,
}

velocity_field_dict = {
    'concat': velocity_field.VelocityField,
}


class Compositional4D(nn.Module):
    ''' Networks for 4D compositional representation.

    Args:
        decoder (nn.Module): Decoder model
        encoder_latent (nn.Module): Latent encoder model
        encoder_temporal (nn.Module): Temporal encoder model
        p0_z (dist): Prior distribution over latent codes z
        device (device): Pytorch device
        input_type (str): Input type
    '''








    # ######################################################
    # #### ODE related functions and helper functions #### #




