def denoising_loop(latents, mask=None, xtrg=None, noise_rescale=None):
    dir_xt = 0
    latents_dtype = latents.dtype
    with self.progress_bar(total=cur_num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            self.attn_state.set_timestep(t.item())
            if (i + skip_t >= mask_start_t and i + skip_t <= mask_end_t and
                xtrg is not None):
                rescale = torch.maximum(1.0 - mask, (1 - mask ** 2) ** 0.5 *
                    inner_strength)
                if noise_rescale is not None:
                    rescale = (1.0 - mask) * (1 - noise_rescale
                        ) + rescale * noise_rescale
                noise = randn_tensor(xtrg.shape, generator=generator,
                    device=device, dtype=xtrg.dtype)
                latents_ref = self.scheduler.add_noise(xtrg, noise, t)
                latents = latents_ref * mask + (1.0 - mask) * (latents - dir_xt
                    ) + rescale * dir_xt
                latents = latents.to(latents_dtype)
            latent_model_input = torch.cat([latents] * 2
                ) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            if guess_mode and do_classifier_free_guidance:
                control_model_input = latents
                control_model_input = self.scheduler.scale_model_input(
                    control_model_input, t)
                controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
            else:
                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds
            if isinstance(controlnet_keep[i], list):
                cond_scale = [(c * s) for c, s in zip(
                    controlnet_conditioning_scale, controlnet_keep[i])]
            else:
                controlnet_cond_scale = controlnet_conditioning_scale
                if isinstance(controlnet_cond_scale, list):
                    controlnet_cond_scale = controlnet_cond_scale[0]
                cond_scale = controlnet_cond_scale * controlnet_keep[i]
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                control_model_input, t, encoder_hidden_states=
                controlnet_prompt_embeds, controlnet_cond=control_image,
                conditioning_scale=cond_scale, guess_mode=guess_mode,
                return_dict=False)
            if guess_mode and do_classifier_free_guidance:
                down_block_res_samples = [torch.cat([torch.zeros_like(d), d
                    ]) for d in down_block_res_samples]
                mid_block_res_sample = torch.cat([torch.zeros_like(
                    mid_block_res_sample), mid_block_res_sample])
            noise_pred = self.unet(latent_model_input, t,
                encoder_hidden_states=prompt_embeds, cross_attention_kwargs
                =cross_attention_kwargs, down_block_additional_residuals=
                down_block_res_samples, mid_block_additional_residual=
                mid_block_res_sample, return_dict=False)[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            pred_x0 = (latents - beta_prod_t ** 0.5 * noise_pred
                ) / alpha_prod_t ** 0.5
            if i + skip_t >= warp_start_t and i + skip_t <= warp_end_t:
                pred_x0 = flow_warp(first_x0_list[i], warp_flow, mode='nearest'
                    ) * warp_mask + (1 - warp_mask) * pred_x0
                latents = self.scheduler.add_noise(pred_x0, noise_pred, t).to(
                    latents_dtype)
            prev_t = (t - self.scheduler.config.num_train_timesteps // self
                .scheduler.num_inference_steps)
            if i == len(timesteps) - 1:
                alpha_t_prev = 1.0
            else:
                alpha_t_prev = self.scheduler.alphas_cumprod[prev_t]
            dir_xt = (1.0 - alpha_t_prev) ** 0.5 * noise_pred
            latents = self.scheduler.step(noise_pred, t, latents, **
                extra_step_kwargs, return_dict=False)[0]
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
        return latents
