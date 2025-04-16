import math
import tempfile
from typing import List, Optional

import numpy as np
import PIL.Image
import torch
from accelerate import Accelerator
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.optimization import get_scheduler


class SdeDragPipeline(DiffusionPipeline):
    r"""
    Pipeline for image drag-and-drop editing using stochastic differential equations: https://arxiv.org/abs/2311.01410.
    Please refer to the [official repository](https://github.com/ML-GSAI/SDE-Drag) for more information.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

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
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Please use
            [`DDIMScheduler`].
    """


    @torch.no_grad()




    @torch.no_grad()


    @torch.no_grad()

    @torch.no_grad()





        for source_, target_ in zip(source_new, target_new):
            r_x_lower, r_x_upper, r_y_lower, r_y_upper = adaption_r(
                source_, target_, adapt_radius, max_height, max_width
            )

            source_feature = latent[
                :, :, source_[1] - r_y_lower : source_[1] + r_y_upper, source_[0] - r_x_lower : source_[0] + r_x_upper
            ].clone()

            latent[
                :, :, source_[1] - r_y_lower : source_[1] + r_y_upper, source_[0] - r_x_lower : source_[0] + r_x_upper
            ] = image_scale * source_feature + noise_scale * torch.randn(
                latent.shape[0],
                4,
                r_y_lower + r_y_upper,
                r_x_lower + r_x_upper,
                device=self.device,
                generator=generator,
            )

            latent[
                :, :, target_[1] - r_y_lower : target_[1] + r_y_upper, target_[0] - r_x_lower : target_[0] + r_x_upper
            ] = source_feature * 1.1
        return latent

    @torch.no_grad()
    def _get_img_latent(self, image, height=None, weight=None):
        data = image.convert("RGB")
        if height is not None:
            data = data.resize((weight, height))
        transform = transforms.ToTensor()
        data = transform(data).unsqueeze(0)
        data = (data * 2.0) - 1.0
        data = data.to(self.device, dtype=self.vae.dtype)
        latent = self.vae.encode(data).latent_dist.sample()
        latent = 0.18215 * latent
        return latent

    @torch.no_grad()
    def _get_eps(self, latent, timestep, guidance_scale, text_embeddings, lora_scale=None):
        latent_model_input = torch.cat([latent] * 2) if guidance_scale > 1.0 else latent
        text_embeddings = text_embeddings if guidance_scale > 1.0 else text_embeddings.chunk(2)[1]

        cross_attention_kwargs = None if lora_scale is None else {"scale": lora_scale}

        with torch.no_grad():
            noise_pred = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample

        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        elif guidance_scale == 1.0:
            noise_pred_text = noise_pred
            noise_pred_uncond = 0.0
        else:
            raise NotImplementedError(guidance_scale)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred

    def _forward_sde(
        self, timestep, sample, guidance_scale, text_embeddings, steps, eta=1.0, lora_scale=None, generator=None
    ):
        num_train_timesteps = len(self.scheduler)
        alphas_cumprod = self.scheduler.alphas_cumprod
        initial_alpha_cumprod = torch.tensor(1.0)

        prev_timestep = timestep + num_train_timesteps // steps

        alpha_prod_t = alphas_cumprod[timestep] if timestep >= 0 else initial_alpha_cumprod
        alpha_prod_t_prev = alphas_cumprod[prev_timestep]

        beta_prod_t_prev = 1 - alpha_prod_t_prev

        x_prev = (alpha_prod_t_prev / alpha_prod_t) ** (0.5) * sample + (1 - alpha_prod_t_prev / alpha_prod_t) ** (
            0.5
        ) * torch.randn(
            sample.size(), dtype=sample.dtype, layout=sample.layout, device=self.device, generator=generator
        )
        eps = self._get_eps(x_prev, prev_timestep, guidance_scale, text_embeddings, lora_scale)

        sigma_t_prev = (
            eta
            * (1 - alpha_prod_t) ** (0.5)
            * (1 - alpha_prod_t_prev / (1 - alpha_prod_t_prev) * (1 - alpha_prod_t) / alpha_prod_t) ** (0.5)
        )

        pred_original_sample = (x_prev - beta_prod_t_prev ** (0.5) * eps) / alpha_prod_t_prev ** (0.5)
        pred_sample_direction_coeff = (1 - alpha_prod_t - sigma_t_prev**2) ** (0.5)

        noise = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample - pred_sample_direction_coeff * eps
        ) / sigma_t_prev

        return x_prev, noise

    def _sample(
        self,
        timestep,
        sample,
        guidance_scale,
        text_embeddings,
        steps,
        sde=False,
        noise=None,
        eta=1.0,
        lora_scale=None,
        generator=None,
    ):
        num_train_timesteps = len(self.scheduler)
        alphas_cumprod = self.scheduler.alphas_cumprod
        final_alpha_cumprod = torch.tensor(1.0)

        eps = self._get_eps(sample, timestep, guidance_scale, text_embeddings, lora_scale)

        prev_timestep = timestep - num_train_timesteps // steps

        alpha_prod_t = alphas_cumprod[timestep]
        alpha_prod_t_prev = alphas_cumprod[prev_timestep] if prev_timestep >= 0 else final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        sigma_t = (
            eta
            * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** (0.5)
            * (1 - alpha_prod_t / alpha_prod_t_prev) ** (0.5)
            if sde
            else 0
        )

        pred_original_sample = (sample - beta_prod_t ** (0.5) * eps) / alpha_prod_t ** (0.5)
        pred_sample_direction_coeff = (1 - alpha_prod_t_prev - sigma_t**2) ** (0.5)

        noise = (
            torch.randn(
                sample.size(), dtype=sample.dtype, layout=sample.layout, device=self.device, generator=generator
            )
            if noise is None
            else noise
        )
        latent = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction_coeff * eps + sigma_t * noise
        )

        return latent

    def _forward(self, latent, steps, t0, lora_scale_min, text_embeddings, generator):

        noises = []
        latents = []
        lora_scales = []
        cfg_scales = []
        latents.append(latent)
        t0 = int(t0 * steps)
        t_begin = steps - t0

        length = len(self.scheduler.timesteps[t_begin - 1 : -1]) - 1
        index = 1
        for t in self.scheduler.timesteps[t_begin:].flip(dims=[0]):
            lora_scale = scale_schedule(1, lora_scale_min, index, length, type="cos")
            cfg_scale = scale_schedule(1, 3.0, index, length, type="linear")
            latent, noise = self._forward_sde(
                t, latent, cfg_scale, text_embeddings, steps, lora_scale=lora_scale, generator=generator
            )

            noises.append(noise)
            latents.append(latent)
            lora_scales.append(lora_scale)
            cfg_scales.append(cfg_scale)
            index += 1
        return latent, noises, latents, lora_scales, cfg_scales

    def _backward(
        self, latent, mask, steps, t0, noises, hook_latents, lora_scales, cfg_scales, text_embeddings, generator
    ):
        t0 = int(t0 * steps)
        t_begin = steps - t0

        hook_latent = hook_latents.pop()
        latent = torch.where(mask > 128, latent, hook_latent)
        for t in self.scheduler.timesteps[t_begin - 1 : -1]:
            latent = self._sample(
                t,
                latent,
                cfg_scales.pop(),
                text_embeddings,
                steps,
                sde=True,
                noise=noises.pop(),
                lora_scale=lora_scales.pop(),
                generator=generator,
            )
            hook_latent = hook_latents.pop()
            latent = torch.where(mask > 128, latent, hook_latent)
        return latent