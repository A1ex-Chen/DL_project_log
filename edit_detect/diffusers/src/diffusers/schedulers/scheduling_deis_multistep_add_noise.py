def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor,
    timesteps: torch.IntTensor) ->torch.Tensor:
    sigmas = self.sigmas.to(device=original_samples.device, dtype=
        original_samples.dtype)
    if original_samples.device.type == 'mps' and torch.is_floating_point(
        timesteps):
        schedule_timesteps = self.timesteps.to(original_samples.device,
            dtype=torch.float32)
        timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
    else:
        schedule_timesteps = self.timesteps.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)
    if self.begin_index is None:
        step_indices = [self.index_for_timestep(t, schedule_timesteps) for
            t in timesteps]
    elif self.step_index is not None:
        step_indices = [self.step_index] * timesteps.shape[0]
    else:
        step_indices = [self.begin_index] * timesteps.shape[0]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(original_samples.shape):
        sigma = sigma.unsqueeze(-1)
    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
    noisy_samples = alpha_t * original_samples + sigma_t * noise
    return noisy_samples
