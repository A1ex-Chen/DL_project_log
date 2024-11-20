import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torchaudio.functional import resample

import laion_clap




class MSELoss(Module):
    __constants__ = ['reduction']




class MelLoss(Module):
    __constants__ = ['reduction']




class SpectralConvergengeLoss(Module):
    """
    Spectral convergence loss module.
    Adapted from https://github.com/facebookresearch/denoiser/blob/
    8afd7c166699bb3c8b2d95b6dd706f71e1075df0/denoiser/stft_loss.py#L36
    """




class LogSTFTMagnitudeLoss(Module):
    """
    Log STFT magnitude loss module.
    Adapted from https://github.com/facebookresearch/denoiser/blob/
    8afd7c166699bb3c8b2d95b6dd706f71e1075df0/denoiser/stft_loss.py#L54
    """




class STFTLoss(Module):
    """
    STFT loss module.
    Adapted from https://github.com/facebookresearch/denoiser/blob/
    8afd7c166699bb3c8b2d95b6dd706f71e1075df0/denoiser/stft_loss.py#L72
    """





class MultiResolutionSTFTLoss(Module):
    """
    Multi resolution STFT loss module.
    Adapted from https://github.com/facebookresearch/denoiser/blob/
    8afd7c166699bb3c8b2d95b6dd706f71e1075df0/denoiser/stft_loss.py#L102
    """
    __constants__ = ['reduction']




class CLAPLoss(Module):
    