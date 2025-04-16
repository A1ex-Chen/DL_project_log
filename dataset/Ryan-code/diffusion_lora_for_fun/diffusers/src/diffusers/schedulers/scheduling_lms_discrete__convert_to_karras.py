def _convert_to_karras(self, in_sigmas: torch.Tensor) ->torch.Tensor:
    """Constructs the noise schedule of Karras et al. (2022)."""
    sigma_min: float = in_sigmas[-1].item()
    sigma_max: float = in_sigmas[0].item()
    rho = 7.0
    ramp = np.linspace(0, 1, self.num_inference_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas
