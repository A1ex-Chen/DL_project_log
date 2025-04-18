def compute_snr(self, timesteps):
    """ Computes the signal-noise ratio (SNR) as per 
            https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/
            521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/
            gaussian_diffusion.py#L847-L849
            Compatible with the DDPM scheduler.
        """
    alphas_cumprod = self.noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(timesteps.device)[timesteps
        ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device
        =timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None
            ]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)
    return (alpha / sigma) ** 2
