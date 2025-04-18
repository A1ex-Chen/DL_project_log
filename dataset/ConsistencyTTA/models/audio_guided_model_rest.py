from copy import deepcopy
from typing import Any, Mapping
from collections import OrderedDict

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger

from diffusers import DDPMScheduler
from diffusers.utils import randn_tensor
from models.audio_distilled_model import AudioDistilledModel

logger = get_logger(__name__, log_level="INFO")


class AudioGDM(AudioDistilledModel):




    @torch.no_grad()


        # Check if the relevant models are in eval mode and frozen
        self.check_eval_mode()

        # Encode text; this is unchanged compared with TANGO
        prompt_embeds_cf, prompt_mask_cf, prompt_embeds, prompt_mask = \
            self.get_prompt_embeds(
                prompt, self.use_teacher_cf_guidance, num_samples_per_prompt=1
            )

        # Sample a random available time step
        t_n = get_random_timestep(z_0.shape[0])

        # Noisy latents
        gaussian_noise = torch.randn_like(z_0)
        z_noisy = self.noise_scheduler.add_noise(z_0, gaussian_noise, t_n)
        z_gaussian = gaussian_noise * self.noise_scheduler.init_noise_sigma

        # Resample the final time step
        last_mask = (t_n == self.noise_scheduler.timesteps.max()).reshape(-1, 1, 1, 1)
        z_n = torch.where(last_mask, z_gaussian.to(z_0.device), z_noisy)
        z_n_scaled = self.noise_scheduler.scale_model_input(z_n, t_n)

        if self.teacher_guidance_scale == -1:  # Random guidance scale
            guidance_scale = torch.rand(z_0.shape[0]) * self.max_rand_guidance_scale
            guidance_scale = guidance_scale.to(z_0.device)
        else:
            guidance_scale = None

        # Query the diffusion teacher model
        with torch.no_grad():
            noise_pred_teacher = self._query_teacher(
                z_n_scaled, t_n, prompt_embeds_cf, prompt_mask_cf, guidance_scale
            )

        noise_pred_student = self.student_unet(
            z_n_scaled, t_n, guidance=guidance_scale,
            encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_mask
        ).sample

        return get_loss(noise_pred_student, noise_pred_teacher, t_n)

    @torch.no_grad()
    def inference(
        self, prompt, inference_scheduler, guidance_scale_input=3, guidance_scale_post=1,
        num_steps=20, use_edm=False, num_samples=1, use_ema=True, query_teacher=False,
        **kwargs
    ):
        self.check_eval_mode()
        device = self.text_encoder.device
        batch_size = len(prompt) * num_samples
        use_cf_guidance = guidance_scale_post > 1.

        prompt_embeds_cf, prompt_mask_cf, prompt_embeds, prompt_mask = \
            self.encode_text_classifier_free(prompt, num_samples)
        encoder_states_stu, encoder_att_mask_stu = \
            (prompt_embeds_cf, prompt_mask_cf) if use_cf_guidance \
                else (prompt_embeds, prompt_mask)
        encoder_states_tea, encoder_att_mask_tea = \
            (prompt_embeds_cf, prompt_mask_cf) if self.use_teacher_cf_guidance \
                else (prompt_embeds, prompt_mask)

        # Query the inference scheduler to obtain the time steps.
        # The time steps uniformly spread between 0 and training time steps (1000).
        # I. e., if num_steps is 3, then the timesteps are 666, 333, and 0.
        inference_scheduler.set_timesteps(num_steps, device=device)
        timesteps = inference_scheduler.timesteps
        if query_teacher:
            inference_scheduler_tea = deepcopy(inference_scheduler)

        # Prepare initial noisy latent
        latent_shape = (batch_size, self.student_unet.config.in_channels, 256, 16)
        zhat_N = randn_tensor(
            latent_shape, generator=None, device=device, dtype=prompt_embeds.dtype
        ) * inference_scheduler.init_noise_sigma
        zhat_n_stu, zhat_n_tea = zhat_N, zhat_N

        # Query the student model
        for t in timesteps:
            # If desired, you can opt to "further guide" the guided student
            zhat_n_input_stu = torch.cat([zhat_n_stu] * 2) if use_cf_guidance else zhat_n_stu
            zhat_n_input_stu = inference_scheduler.scale_model_input(zhat_n_input_stu, t)

            unet = self.student_ema_unet if use_ema else self.student_unet
            noise_pred_stu = unet(
                zhat_n_input_stu, t, guidance=guidance_scale_input,
                encoder_hidden_states=encoder_states_stu,
                encoder_attention_mask=encoder_att_mask_stu
            ).sample

            if use_cf_guidance:
                # Chop the noise prediction into two chunks
                noise_pred_uncond_stu, noise_pred_cond_stu = noise_pred_stu.chunk(2)
                noise_pred_stu = (
                    noise_pred_uncond_stu + 
                    guidance_scale_post * (noise_pred_cond_stu - noise_pred_uncond_stu)
                )

            # compute the previous noisy sample x_t -> x_t-1
            zhat_n_stu = inference_scheduler.step(noise_pred_stu, t, zhat_n_stu).prev_sample

            if query_teacher:  # Query the teacher model as well
                zhat_n_scaled_tea = inference_scheduler.scale_model_input(zhat_n_tea, t)
                noise_pred_tea = self._query_teacher(
                    zhat_n_scaled_tea, t, encoder_states_tea,
                    encoder_att_mask_tea, guidance_scale_input
                )
                zhat_n_tea = inference_scheduler_tea.step(
                    noise_pred_tea, t, zhat_n_tea
                ).prev_sample
                # print(f"Loss w.r.t. teacher: {F.mse_loss(zhat_n_tea, zhat_n_stu):.3f}.")

        if query_teacher:
            logger.info(f"Loss w.r.t. teacher: {F.mse_loss(zhat_n_tea, zhat_n_stu):.3f}.")

        return zhat_n_stu