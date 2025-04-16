@torch.no_grad()
def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
    if use_original_steps:
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
    else:
        sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
        sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas
    if noise is None:
        noise = torch.randn_like(x0)
    return extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape
        ) * x0 + extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape
        ) * noise
