import torch
import torch.nn as nn
from torch import distributions as dist
from lib.oflow.models import (
    encoder_latent, decoder, velocity_field)
from lib.utils.torchdiffeq.torchdiffeq import odeint, odeint_adjoint

encoder_latent_dict = {
    'simple': encoder_latent.Encoder,
    'pointnet': encoder_latent.PointNet,
}

decoder_dict = {
    'simple': decoder.Decoder,
    'cbatchnorm': decoder.DecoderCBatchNorm,
}

velocity_field_dict = {
    'concat': velocity_field.VelocityField,
}


class OccupancyFlow(nn.Module):
    ''' Occupancy Flow model class.

    It consists of a decoder and, depending on the respective settings, an
    encoder, a temporal encoder, an latent encoder, a latent temporal encoder,
    and a vector field.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_latent (nn.Module): latent encoder network
        encoder_latent_temporal (nn.Module): latent temporal encoder network
        encoder_temporal (nn.Module): temporal encoder network
        vector_field (nn.Module): vector field network
        ode_step_size (float): step size of ode solver
        use_adjoint (bool): whether to use the adjoint method for obtaining
            gradients
        rtol (float): relative tolerance for ode solver
        atol (float): absolute tolerance for ode solver
        ode_solver (str): ode solver method
        p0_z (dist): prior distribution
        device (device): PyTorch device
        input_type (str): type of input

    '''








    # ######################################################
    # #### Encoding related functions #### #




    # ######################################################
    # #### Forward and Backward Flow functions #### #




    # ######################################################
    # #### ODE related functions and helper functions #### #



