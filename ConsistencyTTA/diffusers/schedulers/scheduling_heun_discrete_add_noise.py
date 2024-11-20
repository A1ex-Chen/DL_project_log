def add_noise(self, original_samples: torch.FloatTensor, noise: torch.
    FloatTensor, timesteps: torch.FloatTensor) ->torch.FloatTensor:
    self.sigmas = self.sigmas.to(device=original_samples.device, dtype=
        original_samples.dtype)
    self.timesteps = self.timesteps.to(original_samples.device)
    timesteps = timesteps.to(original_samples.device)
    step_indices = self.index_for_timestep(timesteps)
    sigma = self.sigmas[step_indices].flatten()
    while len(sigma.shape) < len(original_samples.shape):
        sigma = sigma.unsqueeze(-1)
    noisy_samples = original_samples + noise * sigma
    return noisy_samples
