def _compute_exponential_sigmas(self, ramp, sigma_min=None, sigma_max=None
    ) ->torch.Tensor:
    """Implementation closely follows k-diffusion.

        https://github.com/crowsonkb/k-diffusion/blob/6ab5146d4a5ef63901326489f31f1d8e7dd36b48/k_diffusion/sampling.py#L26
        """
    sigma_min = sigma_min or self.config.sigma_min
    sigma_max = sigma_max or self.config.sigma_max
    sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max), len(ramp)
        ).exp().flip(0)
    return sigmas
