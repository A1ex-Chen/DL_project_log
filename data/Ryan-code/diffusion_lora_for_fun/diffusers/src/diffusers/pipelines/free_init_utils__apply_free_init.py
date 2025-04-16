def _apply_free_init(self, latents: torch.Tensor, free_init_iteration: int,
    num_inference_steps: int, device: torch.device, dtype: torch.dtype,
    generator: torch.Generator):
    if free_init_iteration == 0:
        self._free_init_initial_noise = latents.detach().clone()
    else:
        latent_shape = latents.shape
        free_init_filter_shape = 1, *latent_shape[1:]
        free_init_freq_filter = self._get_free_init_freq_filter(shape=
            free_init_filter_shape, device=device, filter_type=self.
            _free_init_method, order=self._free_init_order,
            spatial_stop_frequency=self._free_init_spatial_stop_frequency,
            temporal_stop_frequency=self._free_init_temporal_stop_frequency)
        current_diffuse_timestep = (self.scheduler.config.
            num_train_timesteps - 1)
        diffuse_timesteps = torch.full((latent_shape[0],),
            current_diffuse_timestep).long()
        z_t = self.scheduler.add_noise(original_samples=latents, noise=self
            ._free_init_initial_noise, timesteps=diffuse_timesteps.to(device)
            ).to(dtype=torch.float32)
        z_rand = randn_tensor(shape=latent_shape, generator=generator,
            device=device, dtype=torch.float32)
        latents = self._apply_freq_filter(z_t, z_rand, low_pass_filter=
            free_init_freq_filter)
        latents = latents.to(dtype)
    if self._free_init_use_fast_sampling:
        num_inference_steps = max(1, int(num_inference_steps / self.
            _free_init_num_iters * (free_init_iteration + 1)))
        self.scheduler.set_timesteps(num_inference_steps, device=device)
    return latents, self.scheduler.timesteps
