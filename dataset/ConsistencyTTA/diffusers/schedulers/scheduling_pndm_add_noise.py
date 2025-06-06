def add_noise(self, original_samples: torch.FloatTensor, noise: torch.
    FloatTensor, timesteps: torch.IntTensor) ->torch.Tensor:
    self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.
        device, dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)
    sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    noisy_samples = (sqrt_alpha_prod * original_samples + 
        sqrt_one_minus_alpha_prod * noise)
    return noisy_samples
