# Copyright 2024 Bingxin Ke, ETH Zurich and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import logging
import math
from typing import Dict, Union

import matplotlib
import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
from scipy.optimize import minimize
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput, check_min_version


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0")


class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`None` or `PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: np.ndarray
    depth_colored: Union[None, Image.Image]
    uncertainty: Union[None, np.ndarray]




class MarigoldPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215


    @torch.no_grad()



    @torch.no_grad()



    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

        device = input_images.device
        dtype = input_images.dtype
        np_dtype = np.float32

        original_input = input_images.clone()
        n_img = input_images.shape[0]
        ori_shape = input_images.shape

        if max_res is not None:
            scale_factor = torch.min(max_res / torch.tensor(ori_shape[-2:]))
            if scale_factor < 1:
                downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
                input_images = downscaler(torch.from_numpy(input_images)).numpy()

        # init guess
        _min = np.min(input_images.reshape((n_img, -1)).cpu().numpy(), axis=1)
        _max = np.max(input_images.reshape((n_img, -1)).cpu().numpy(), axis=1)
        s_init = 1.0 / (_max - _min).reshape((-1, 1, 1))
        t_init = (-1 * s_init.flatten() * _min.flatten()).reshape((-1, 1, 1))
        x = np.concatenate([s_init, t_init]).reshape(-1).astype(np_dtype)

        input_images = input_images.to(device)

        # objective function

        res = minimize(
            closure,
            x,
            method="BFGS",
            tol=tol,
            options={"maxiter": max_iter, "disp": False},
        )
        x = res.x
        l = len(x)
        s = x[: int(l / 2)]
        t = x[int(l / 2) :]

        # Prediction
        s = torch.from_numpy(s).to(dtype=dtype).to(device)
        t = torch.from_numpy(t).to(dtype=dtype).to(device)
        transformed_arrays = original_input * s.view(-1, 1, 1) + t.view(-1, 1, 1)
        if "mean" == reduction:
            aligned_images = torch.mean(transformed_arrays, dim=0)
            std = torch.std(transformed_arrays, dim=0)
            uncertainty = std
        elif "median" == reduction:
            aligned_images = torch.median(transformed_arrays, dim=0).values
            # MAD (median absolute deviation) as uncertainty indicator
            abs_dev = torch.abs(transformed_arrays - aligned_images)
            mad = torch.median(abs_dev, dim=0).values
            uncertainty = mad
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")

        # Scale and shift to [0, 1]
        _min = torch.min(aligned_images)
        _max = torch.max(aligned_images)
        aligned_images = (aligned_images - _min) / (_max - _min)
        uncertainty /= _max - _min

        return aligned_images, uncertainty