def add_noise(self, original_samples: torch.FloatTensor, noise: torch.
    FloatTensor, timesteps: torch.FloatTensor) ->torch.FloatTensor:
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
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in
        timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(original_samples.shape):
        sigma = sigma.unsqueeze(-1)
    noisy_samples = original_samples + noise * sigma
    return noisy_samples
