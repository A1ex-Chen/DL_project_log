def get_input_example(self, prompt, height=None, width=None, guidance_scale
    =7.5, num_images_per_prompt=1):
    prompt_embeds = None
    negative_prompt_embeds = None
    negative_prompt = None
    callback_steps = 1
    generator = None
    latents = None
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    self.check_inputs(prompt, height, width, callback_steps,
        negative_prompt, prompt_embeds, negative_prompt_embeds)
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    device = 'cpu'
    do_classifier_free_guidance = guidance_scale > 1.0
    prompt_embeds = self._encode_prompt(prompt, device,
        num_images_per_prompt, do_classifier_free_guidance, negative_prompt,
        prompt_embeds=prompt_embeds, negative_prompt_embeds=
        negative_prompt_embeds)
    latents = self.prepare_latents(batch_size * num_images_per_prompt, self
        .unet.config.in_channels, height, width, prompt_embeds.dtype,
        device, generator, latents)
    dummy = torch.ones(1, dtype=torch.int32)
    latent_model_input = torch.cat([latents] * 2
        ) if do_classifier_free_guidance else latents
    latent_model_input = self.scheduler.scale_model_input(latent_model_input,
        dummy)
    unet_input_example = latent_model_input, dummy, prompt_embeds
    vae_decoder_input_example = latents
    return unet_input_example, vae_decoder_input_example
