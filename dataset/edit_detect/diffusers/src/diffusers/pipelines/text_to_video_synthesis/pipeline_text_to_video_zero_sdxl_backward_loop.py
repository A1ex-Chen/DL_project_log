def backward_loop(self, latents, timesteps, prompt_embeds, guidance_scale,
    callback, callback_steps, num_warmup_steps, extra_step_kwargs,
    add_text_embeds, add_time_ids, cross_attention_kwargs=None,
    guidance_rescale: float=0.0):
    """
        Perform backward process given list of time steps

        Args:
            latents:
                Latents at time timesteps[0].
            timesteps:
                Time steps along which to perform backward process.
            prompt_embeds:
                Pre-generated text embeddings.
            guidance_scale:
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            extra_step_kwargs:
                Extra_step_kwargs.
            cross_attention_kwargs:
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            num_warmup_steps:
                number of warmup steps.

        Returns:
            latents: latents of backward process output at time timesteps[-1]
        """
    do_classifier_free_guidance = guidance_scale > 1.0
    num_steps = (len(timesteps) - num_warmup_steps) // self.scheduler.order
    with self.progress_bar(total=num_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2
                ) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            added_cond_kwargs = {'text_embeds': add_text_embeds, 'time_ids':
                add_time_ids}
            noise_pred = self.unet(latent_model_input, t,
                encoder_hidden_states=prompt_embeds, cross_attention_kwargs
                =cross_attention_kwargs, added_cond_kwargs=
                added_cond_kwargs, return_dict=False)[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            if do_classifier_free_guidance and guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text,
                    guidance_rescale=guidance_rescale)
            latents = self.scheduler.step(noise_pred, t, latents, **
                extra_step_kwargs).prev_sample
            if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1
                ) % self.scheduler.order == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
    return latents.clone().detach()
