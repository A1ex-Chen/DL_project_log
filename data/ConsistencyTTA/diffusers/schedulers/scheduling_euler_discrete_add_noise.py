def add_noise(self, original_samples: torch.FloatTensor, noise: torch.
    FloatTensor, timesteps: torch.FloatTensor) ->torch.FloatTensor:
    self.sigmas = self.sigmas.to(device=original_samples.device, dtype=
        original_samples.dtype)
    if original_samples.device.type == 'mps' and torch.is_floating_point(
        timesteps):
        self.timesteps = self.timesteps.to(original_samples.device, dtype=
            torch.float32)
        timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
    else:
        self.timesteps = self.timesteps.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)
    schedule_timesteps = self.timesteps
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in
        timesteps]
    sigma = self.sigmas[step_indices].flatten()
    while len(sigma.shape) < len(original_samples.shape):
        sigma = sigma.unsqueeze(-1)
    noisy_samples = original_samples + noise * sigma
    return noisy_samples
