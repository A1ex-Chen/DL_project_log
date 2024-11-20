def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor,
    timesteps: torch.Tensor) ->torch.Tensor:
    timesteps = timesteps.to(original_samples.device)
    sigmas = self.discrete_sigmas.to(original_samples.device)[timesteps]
    noise = noise * sigmas[:, None, None, None
        ] if noise is not None else torch.randn_like(original_samples
        ) * sigmas[:, None, None, None]
    noisy_samples = noise + original_samples
    return noisy_samples
