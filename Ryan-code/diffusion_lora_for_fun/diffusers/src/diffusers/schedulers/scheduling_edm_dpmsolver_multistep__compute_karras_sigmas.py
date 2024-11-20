def _compute_karras_sigmas(self, ramp, sigma_min=None, sigma_max=None
    ) ->torch.Tensor:
    """Constructs the noise schedule of Karras et al. (2022)."""
    sigma_min = sigma_min or self.config.sigma_min
    sigma_max = sigma_max or self.config.sigma_max
    rho = self.config.rho
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas
