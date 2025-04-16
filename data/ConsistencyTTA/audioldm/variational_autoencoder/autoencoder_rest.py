import torch
from torch import nn
from einops import rearrange

from audioldm.variational_autoencoder.modules import Encoder, Decoder
from audioldm.variational_autoencoder.distributions import DiagonalGaussianDistribution
from audioldm.hifigan.utilities import get_vocoder, vocoder_infer


class AutoencoderKL(nn.Module):

    @property




    @torch.no_grad()




