def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor,
    timesteps: torch.IntTensor) ->torch.Tensor:
    betas = torch.linspace(0.0001, 0.02, 1000, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = alphas_cumprod.to(device=original_samples.device,
        dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    noisy_samples = (sqrt_alpha_prod * original_samples + 
        sqrt_one_minus_alpha_prod * noise)
    return noisy_samples
