def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps
    ) ->torch.Tensor:
    """Constructs the noise schedule of Karras et al. (2022)."""
    if hasattr(self.config, 'sigma_min'):
        sigma_min = self.config.sigma_min
    else:
        sigma_min = None
    if hasattr(self.config, 'sigma_max'):
        sigma_max = self.config.sigma_max
    else:
        sigma_max = None
    sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
    sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()
    rho = 7.0
    ramp = np.linspace(0, 1, num_inference_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas
