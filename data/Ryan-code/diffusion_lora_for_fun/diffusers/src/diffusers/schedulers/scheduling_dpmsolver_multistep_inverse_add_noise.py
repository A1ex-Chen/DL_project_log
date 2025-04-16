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
    step_indices = []
    for timestep in timesteps:
        index_candidates = (schedule_timesteps == timestep).nonzero()
        if len(index_candidates) == 0:
            step_index = len(schedule_timesteps) - 1
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            step_index = index_candidates[0].item()
        step_indices.append(step_index)
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(original_samples.shape):
        sigma = sigma.unsqueeze(-1)
    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
    noisy_samples = alpha_t * original_samples + sigma_t * noise
    return noisy_samples
