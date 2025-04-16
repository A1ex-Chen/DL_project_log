def __denoise_latent(self, latents, text_embeddings, timesteps=None,
    step_offset=0, mask=None, masked_image_latents=None):
    if not isinstance(timesteps, torch.Tensor):
        timesteps = self.scheduler.timesteps
    for step_index, timestep in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = self.scheduler.scale_model_input(
            latent_model_input, timestep)
        if isinstance(mask, torch.Tensor):
            latent_model_input = torch.cat([latent_model_input, mask,
                masked_image_latents], dim=1)
        timestep_float = timestep.float(
            ) if timestep.dtype != torch.float32 else timestep
        sample_inp = device_view(latent_model_input)
        timestep_inp = device_view(timestep_float)
        embeddings_inp = device_view(text_embeddings)
        noise_pred = runEngine(self.engine['unet'], {'sample': sample_inp,
            'timestep': timestep_inp, 'encoder_hidden_states':
            embeddings_inp}, self.stream)['latent']
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text
             - noise_pred_uncond)
        latents = self.scheduler.step(noise_pred, timestep, latents
            ).prev_sample
    latents = 1.0 / 0.18215 * latents
    return latents
