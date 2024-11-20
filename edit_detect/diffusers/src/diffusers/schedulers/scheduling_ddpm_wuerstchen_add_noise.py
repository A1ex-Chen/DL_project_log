def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor,
    timesteps: torch.Tensor) ->torch.Tensor:
    device = original_samples.device
    dtype = original_samples.dtype
    alpha_cumprod = self._alpha_cumprod(timesteps, device=device).view(
        timesteps.size(0), *[(1) for _ in original_samples.shape[1:]])
    noisy_samples = alpha_cumprod.sqrt() * original_samples + (1 -
        alpha_cumprod).sqrt() * noise
    return noisy_samples.to(dtype=dtype)
