def _convert_to_lu(self, in_lambdas: torch.Tensor, num_inference_steps
    ) ->torch.Tensor:
    """Constructs the noise schedule of Lu et al. (2022)."""
    lambda_min: float = in_lambdas[-1].item()
    lambda_max: float = in_lambdas[0].item()
    rho = 1.0
    ramp = np.linspace(0, 1, num_inference_steps)
    min_inv_rho = lambda_min ** (1 / rho)
    max_inv_rho = lambda_max ** (1 / rho)
    lambdas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return lambdas
