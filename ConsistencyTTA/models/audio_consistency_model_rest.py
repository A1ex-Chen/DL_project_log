from copy import deepcopy
from typing import Any, Mapping
from collections import OrderedDict
from time import time

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger

from diffusers.utils import randn_tensor
from diffusers import DDIMScheduler, HeunDiscreteScheduler
from models.audio_distilled_model import AudioDistilledModel
from tools.train_utils import do_ema_update
from tools.losses import MSELoss, MelLoss, CLAPLoss, MultiResolutionSTFTLoss

logger = get_logger(__name__, log_level="INFO")


class AudioLCM(AudioDistilledModel):










    @torch.no_grad()


        # Check if the relevant models are in eval mode and frozen
        self.check_eval_mode()
        assert validation_mode >= 0

        # Encode text; this is unchanged compared with TANGO
        prompt_embeds_cf, prompt_mask_cf, prompt_embeds, prompt_mask = \
            self.get_prompt_embeds(
                prompt, self.use_teacher_cf_guidance, num_samples_per_prompt=1
            )

        # Randomly mask 10% of the encoder hidden states
        if self.uncondition:
            raise NotImplementedError

        # Sample a random available time step
        t_nplus1, t_n, t_ind_nplus1, t_ind_n = get_random_timestep(
            z_0.shape[0], validation_mode
        )

        # Noisy latents
        gaussian_noise = torch.randn_like(z_0)
        z_noisy = self.noise_scheduler.add_noise(z_0, gaussian_noise, t_nplus1)
        z_gaussian = gaussian_noise * self.noise_scheduler.init_noise_sigma

        # Resample the final time step
        last_step = self.noise_scheduler.timesteps.max()
        last_mask = (t_nplus1 == last_step).reshape(-1, 1, 1, 1)
        z_nplus1 = torch.where(last_mask, z_gaussian.to(z_0.device), z_noisy)
        z_nplus1_scaled = self.noise_scheduler.scale_model_input(z_nplus1, t_nplus1)

        if self.use_edm:
            assert self.noise_scheduler.state_in_first_order

        if self.teacher_guidance_scale == -1:  # Random guidance scale
            guidance_scale = torch.rand(z_0.shape[0]) * self.max_rand_guidance_scale
            guidance_scale = guidance_scale.to(z_0.device) 
        else:
            guidance_scale = None

        # Query the diffusion teacher model
        with torch.no_grad():
            noise_pred_nplus1 = self._query_teacher(
                z_nplus1_scaled, t_nplus1, prompt_embeds_cf, prompt_mask_cf, guidance_scale
            )
            # Recover estimation of z_n from the noise prediction
            zhat_n = self.noise_scheduler.step(
                noise_pred_nplus1, t_nplus1, z_nplus1
            ).prev_sample
            zhat_n_scaled = self.noise_scheduler.scale_model_input(zhat_n, t_n)
            assert not zhat_n_scaled.isnan().any(), f"zhat_n is NaN at t={t_nplus1}"

            if self.use_edm:  # EDM requires two teacher queries
                noise_pred_n = self._query_teacher(
                    zhat_n_scaled, t_n, prompt_embeds_cf, prompt_mask_cf, guidance_scale
                )
                # Step scheduler again to perform Heun update
                zhat_n = self.noise_scheduler.step(noise_pred_n, t_n, zhat_n).prev_sample
                zhat_n_scaled = self.noise_scheduler.scale_model_input(zhat_n, t_n)
                assert not zhat_n_scaled.isnan().any(), f"zhat_n is NaN at t={t_nplus1}"
                assert self.noise_scheduler.state_in_first_order

        # Query the diffusion model to obtain the estimation of z_n
        if validation_mode != 0:
            with torch.no_grad():
                zhat_0_from_nplus1 = self.student_target_unet(
                    z_nplus1_scaled, t_nplus1, guidance=guidance_scale,
                    encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_mask
                ).sample

                zhat_0_from_n = self.student_target_unet(
                    zhat_n_scaled, t_n, guidance=guidance_scale,
                    encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_mask
                ).sample

                if run_teacher:
                    device = self.text_encoder.device
                    avail_timesteps = self.noise_scheduler.timesteps.to(device)

                    for t in avail_timesteps[t_ind_n[0]:]:
                        # Get noise prediction from the diffusion model
                        zhat_n_scaled_tea = self.noise_scheduler.scale_model_input(zhat_n, t)
                        noise_pred_n = self._query_teacher(
                            zhat_n_scaled_tea, t, prompt_embeds_cf, prompt_mask_cf, guidance_scale
                        )
                        # Step scheduler
                        zhat_n = self.noise_scheduler.step(noise_pred_n, t, zhat_n)
                        zhat_n = zhat_n.prev_sample
                        assert not zhat_n.isnan().any()

                    # logger.info(f"loss w/ gt: {F.mse_loss(zhat_0_from_nplus1, z_0).item()}")
                    # logger.info(
                    #     f"loss w/ teacher: {F.mse_loss(zhat_0_from_nplus1, zhat_n).item()}"
                    # )
                    # loss_cons_ = get_loss(
                    #     zhat_0_from_nplus1, zhat_0_from_n, 
                    #     avail_timesteps[t_ind_nplus1[0]], t_ind_nplus1[0]
                    # ).item()
                    # logger.info(f"consistency loss: {loss_cons_}")
                    # logger.info(f"teacher loss: {F.mse_loss(zhat_0_from_n, zhat_n).item()}")

                    if self.use_edm:
                        self.noise_scheduler.prev_derivative = None
                        self.noise_scheduler.dt = None
                        self.noise_scheduler.sample = None

            t_nplus1 = avail_timesteps[t_ind_nplus1[0]]
            loss_w_gt = F.mse_loss(zhat_0_from_nplus1, z_0)  # w.r.t. ground truth
            loss_w_teacher = F.mse_loss(zhat_0_from_nplus1, zhat_n)  # w.r.t. teacher model
            loss_consis = get_loss(
                zhat_0_from_nplus1, zhat_0_from_n, gt_wav, prompt, t_nplus1, t_ind_nplus1[0]
            )
            loss_teacher = F.mse_loss(zhat_n, z_0)  # teacher loss

            return loss_w_gt, loss_w_teacher, loss_consis, loss_teacher

        else:  # Training mode

            with torch.no_grad():
                # Feed both z_n and z_{n+1} into the consistency model and minimize the loss
                zhat_0_from_n = self.student_target_unet(
                    zhat_n_scaled, t_n, guidance=guidance_scale,
                    encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_mask
                ).sample.detach()
                # If t_n is 0, use ground truth latent as the target
                zhat_0_from_n = torch.where(
                    (t_n == 0).reshape(-1, 1, 1, 1), z_0, zhat_0_from_n
                )

            zhat_0_from_nplus1 = self.student_unet(
                z_nplus1_scaled, t_nplus1, guidance=guidance_scale,
                encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_mask
            ).sample

            return get_loss(
                zhat_0_from_nplus1, zhat_0_from_n, gt_wav, prompt, t_nplus1, t_ind_nplus1
            )

    @torch.no_grad()
    def inference(
        self, prompt, inference_scheduler, guidance_scale_input=3, guidance_scale_post=1,
        num_steps=20, use_edm=False, num_samples=1, use_ema=True, 
        query_teacher=False, num_teacher_steps=18, return_all=False
    ):

        self.check_eval_mode()
        device = self.text_encoder.device
        batch_size = len(prompt) * num_samples
        use_cf_guidance = guidance_scale_post > 1.

        # Get prompt embeddings
        t_start_embed = time()
        prompt_embeds_cf, prompt_mask_cf, prompt_embeds, prompt_mask = \
            self.encode_text_classifier_free(prompt, num_samples)
        encoder_states_stu, encoder_att_mask_stu = \
            (prompt_embeds_cf, prompt_mask_cf) if use_cf_guidance \
                else (prompt_embeds, prompt_mask)
        encoder_states_tea, encoder_att_mask_tea = \
            (prompt_embeds_cf, prompt_mask_cf) if self.use_teacher_cf_guidance \
                else (prompt_embeds, prompt_mask)

        # Prepare noise
        num_channels_latents = self.student_target_unet.config.in_channels
        latent_shape = (batch_size, num_channels_latents, 256, 16)
        noise = randn_tensor(
            latent_shape, generator=None, device=device, dtype=prompt_embeds.dtype
        )
        time_embed = time() - t_start_embed

        # Query the inference scheduler to obtain the time steps.
        # The time steps spread between 0 and training time steps
        t_start_stu = time()
        inference_scheduler.set_timesteps(18, device=device)
        z_N_stu = noise * inference_scheduler.init_noise_sigma

        # Query the consistency model
        zhat_0_stu = calc_zhat_0(
            z_N_stu, inference_scheduler.timesteps[0], encoder_states_stu,
            encoder_att_mask_stu, guidance_scale_input, guidance_scale_post
        )

        # Iteratively query the consistency model if requested
        inference_scheduler.set_timesteps(num_steps, device=device)
        order = 2 if self.use_edm else 1

        for t in inference_scheduler.timesteps[1::order]:
            zhat_n_stu = inference_scheduler.add_noise(
                zhat_0_stu, torch.randn_like(zhat_0_stu), t
            )
            # Calculate new zhat_0
            zhat_0_stu = calc_zhat_0(
                zhat_n_stu, t, encoder_states_stu, encoder_att_mask_stu,
                guidance_scale_input, guidance_scale_post
            )
        time_stu = time() - t_start_stu
        if return_all:
            print("Distilled model generation completed!")

        # Query the teacher model as well if requested by user
        if query_teacher:
            t_start_tea = time()
            inference_scheduler.set_timesteps(num_teacher_steps, device=device)
            zhat_n_tea = noise * inference_scheduler.init_noise_sigma

            for t in inference_scheduler.timesteps:
                zhat_n_input = inference_scheduler.scale_model_input(zhat_n_tea, t)
                noise_pred = self._query_teacher(
                    zhat_n_input, t, encoder_states_tea, encoder_att_mask_tea,
                    guidance_scale_input
                )
                zhat_n_tea = inference_scheduler.step(noise_pred, t, zhat_n_tea).prev_sample

            # Reset solver
            if self.use_edm:
                inference_scheduler.prev_derivative = None
                inference_scheduler.dt = None
                inference_scheduler.sample = None

            # loss_w_teacher = ((zhat_0_stu - zhat_n_tea) ** 2).mean().item()
            # logger.info(f"loss w.r.t. teacher: {loss_w_teacher}")
            time_tea = time() - t_start_tea
            if return_all:
                print("Diffusion model generation completed!")

        else:
            zhat_n_tea, time_tea = None, None

        if return_all:
            # Return student generation, teacher generation, student time, teacher time
            if time_tea is not None:
                time_tea += time_embed
            return zhat_0_stu, zhat_n_tea, time_stu + time_embed, time_tea
        else:
            # Return student generation
            return zhat_0_stu