def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor,
    timesteps: torch.Tensor) ->torch.Tensor:
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
    noisy_samples = original_samples + noise * sigma
    return noisy_samples
