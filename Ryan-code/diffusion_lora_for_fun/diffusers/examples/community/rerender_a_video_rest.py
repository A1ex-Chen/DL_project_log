# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from gmflow.gmflow import GMFlow
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import BaseOutput, deprecate, logging
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name










@torch.no_grad()


blur = T.GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))


@dataclass
class TextToVideoSDPipelineOutput(BaseOutput):
    """
    Output class for text-to-video pipelines.

    Args:
        frames (`List[np.ndarray]` or `torch.Tensor`)
            List of denoised frames (essentially images) as NumPy arrays of shape `(height, width, num_channels)` or as
            a `torch` tensor. The length of the list denotes the video length (the number of frames).
    """

    frames: Union[List[np.ndarray], torch.Tensor]


@torch.no_grad()


class AttnState:
    STORE = 0
    LOAD = 1
    LOAD_AND_STORE_PREV = 2

    def __init__(self):
        self.reset()

    @property
    def state(self):
        return self.__state

    @property
    def timestep(self):
        return self.__timestep

    def set_timestep(self, t):
        self.__timestep = t

    def reset(self):
        self.__state = AttnState.STORE
        self.__timestep = 0

    def to_load(self):
        self.__state = AttnState.LOAD

    def to_load_and_store_prev(self):
        self.__state = AttnState.LOAD_AND_STORE_PREV


class CrossFrameAttnProcessor(AttnProcessor):
    """
    Cross frame attention processor. Each frame attends the first frame and previous frame.

    Args:
        attn_state: Whether the model is processing the first frame or an intermediate frame
    """

    def __init__(self, attn_state: AttnState):
        super().__init__()
        self.attn_state = attn_state
        self.first_maps = {}
        self.prev_maps = {}

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        # Is self attention
        if encoder_hidden_states is None:
            t = self.attn_state.timestep
            if self.attn_state.state == AttnState.STORE:
                self.first_maps[t] = hidden_states.detach()
                self.prev_maps[t] = hidden_states.detach()
                res = super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            else:
                if self.attn_state.state == AttnState.LOAD_AND_STORE_PREV:
                    tmp = hidden_states.detach()
                cross_map = torch.cat((self.first_maps[t], self.prev_maps[t]), dim=1)
                res = super().__call__(attn, hidden_states, cross_map, attention_mask, temb)
                if self.attn_state.state == AttnState.LOAD_AND_STORE_PREV:
                    self.prev_maps[t] = tmp
        else:
            res = super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, temb)

        return res



    @property

    @property






class CrossFrameAttnProcessor(AttnProcessor):
    """
    Cross frame attention processor. Each frame attends the first frame and previous frame.

    Args:
        attn_state: Whether the model is processing the first frame or an intermediate frame
    """




def prepare_image(image):
    if isinstance(image, torch.Tensor):
        # Batch single image
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image


class RerenderAVideoPipeline(StableDiffusionControlNetImg2ImgPipeline):
    r"""
    Pipeline for video-to-video translation using Stable Diffusion with Rerender Algorithm.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *Textual-Inversion*: [`loaders.TextualInversionLoaderMixin.load_textual_inversion`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. If you set multiple ControlNets
            as a list, the outputs from each ControlNet are added together to create one combined additional
            conditioning.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    _optional_components = ["safety_checker", "feature_extractor"]


    # Modified from src/diffusers/pipelines/controlnet/pipeline_controlnet.StableDiffusionControlNetImg2ImgPipeline.check_inputs

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.prepare_latents

    @torch.no_grad()


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""




            if mask_start_t <= mask_end_t:
                self.attn_state.to_load()
            else:
                self.attn_state.to_load_and_store_prev()
            latents = denoising_loop(init_latents)

            if mask_start_t <= mask_end_t:
                direct_result = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

                blend_results = (1 - blend_mask_pre) * warped_pre + blend_mask_pre * direct_result
                blend_results = (1 - blend_mask_0) * warped_0 + blend_mask_0 * blend_results

                bwd_occ = 1 - torch.clamp(1 - bwd_occ_pre + 1 - bwd_occ_0, 0, 1)
                blend_mask = blur(F.max_pool2d(bwd_occ, kernel_size=9, stride=1, padding=4))
                blend_mask = 1 - torch.clamp(blend_mask + bwd_occ, 0, 1)

                blend_results = blend_results.to(latents.dtype)
                xtrg = self.vae.encode(blend_results).latent_dist.sample(generator)
                xtrg = self.vae.config.scaling_factor * xtrg
                blend_results_rec = self.vae.decode(xtrg / self.vae.config.scaling_factor, return_dict=False)[0]
                xtrg_rec = self.vae.encode(blend_results_rec).latent_dist.sample(generator)
                xtrg_rec = self.vae.config.scaling_factor * xtrg_rec
                xtrg_ = xtrg + (xtrg - xtrg_rec)
                blend_results_rec_new = self.vae.decode(xtrg_ / self.vae.config.scaling_factor, return_dict=False)[0]
                tmp = (abs(blend_results_rec_new - blend_results).mean(dim=1, keepdims=True) > 0.25).float()

                mask_x = F.max_pool2d(
                    (F.interpolate(tmp, scale_factor=1 / 8.0, mode="bilinear") > 0).float(),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )

                mask = 1 - F.max_pool2d(1 - blend_mask, kernel_size=8)  # * (1-mask_x)

                if smooth_boundary:
                    noise_rescale = find_flat_region(mask)
                else:
                    noise_rescale = torch.ones_like(mask)

                xtrg = (xtrg + (1 - mask_x) * (xtrg - xtrg_rec)) * mask
                xtrg = xtrg.to(latents.dtype)

                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps, cur_num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

                self.attn_state.to_load_and_store_prev()
                latents = denoising_loop(init_latents, mask * mask_strength, xtrg, noise_rescale)

            if not output_type == "latent":
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            else:
                image = latents

            prev_result = image

            do_denormalize = [True] * image.shape[0]
            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

            output_frames.append(image[0])

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return output_frames

        return TextToVideoSDPipelineOutput(frames=output_frames)


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel", padding_factor=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding_factor) + 1) * padding_factor - self.ht) % padding_factor
        pad_wd = (((self.wd // padding_factor) + 1) * padding_factor - self.wd) % padding_factor
        if mode == "sintel":
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]