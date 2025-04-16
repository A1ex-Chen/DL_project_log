def _convert_to_karras(self, ramp):
    """Constructs the noise schedule of Karras et al. (2022)."""
    sigma_min: float = self.config.sigma_min
    sigma_max: float = self.config.sigma_max
    rho = self.config.rho
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas
