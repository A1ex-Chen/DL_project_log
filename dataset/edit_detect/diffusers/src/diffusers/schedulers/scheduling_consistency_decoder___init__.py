@register_to_config
def __init__(self, num_train_timesteps: int=1024, sigma_data: float=0.5):
    betas = betas_for_alpha_bar(num_train_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sigmas = torch.sqrt(1.0 / alphas_cumprod - 1)
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    self.c_skip = sqrt_recip_alphas_cumprod * sigma_data ** 2 / (sigmas ** 
        2 + sigma_data ** 2)
    self.c_out = sigmas * sigma_data / (sigmas ** 2 + sigma_data ** 2) ** 0.5
    self.c_in = sqrt_recip_alphas_cumprod / (sigmas ** 2 + sigma_data ** 2
        ) ** 0.5
