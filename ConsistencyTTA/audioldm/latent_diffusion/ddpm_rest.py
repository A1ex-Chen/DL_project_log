"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import torch
import torch.nn as nn

import numpy as np
from contextlib import contextmanager
from functools import partial
import os
from tqdm import tqdm

from audioldm.utils import exists, default, count_params, instantiate_from_config
from audioldm.latent_diffusion.ema import LitEma
from audioldm.latent_diffusion.util import (
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
)


__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}






class DiffusionWrapper(nn.Module):



class DDPM(nn.Module):
    # classic DDPM with Gaussian diffusion, in image space




    @contextmanager





    @torch.no_grad()

    @torch.no_grad()

    @torch.no_grad()


