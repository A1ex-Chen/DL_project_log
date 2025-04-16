def dpm_solver_first_order_update(self, model_output: torch.Tensor, sample:
    torch.Tensor=None, noise: Optional[torch.Tensor]=None) ->torch.Tensor:
    """
        One step for the first-order DPMSolver (equivalent to DDIM).

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
    sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.
        step_index]
    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
    alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
    h = lambda_t - lambda_s
    if self.config.algorithm_type == 'dpmsolver++':
        x_t = sigma_t / sigma_s * sample - alpha_t * (torch.exp(-h) - 1.0
            ) * model_output
    elif self.config.algorithm_type == 'sde-dpmsolver++':
        assert noise is not None
        x_t = sigma_t / sigma_s * torch.exp(-h) * sample + alpha_t * (1 -
            torch.exp(-2.0 * h)) * model_output + sigma_t * torch.sqrt(1.0 -
            torch.exp(-2 * h)) * noise
    return x_t
