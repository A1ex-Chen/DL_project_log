def add_noise(self, original_samples: torch.FloatTensor, noise: torch.
    FloatTensor, timesteps: torch.FloatTensor) ->torch.FloatTensor:
    timesteps = timesteps.to(original_samples.device)
    sigmas = self.discrete_sigmas.to(original_samples.device)[timesteps]
    noise = torch.randn_like(original_samples) * sigmas[:, None, None, None]
    noisy_samples = noise + original_samples
    return noisy_samples
