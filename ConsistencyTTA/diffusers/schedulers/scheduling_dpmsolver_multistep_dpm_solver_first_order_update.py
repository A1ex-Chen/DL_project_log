def dpm_solver_first_order_update(self, model_output: torch.FloatTensor,
    timestep: int, prev_timestep: int, sample: torch.FloatTensor, noise:
    Optional[torch.FloatTensor]=None) ->torch.FloatTensor:
    """
        One step for the first-order DPM-Solver (equivalent to DDIM).

        See https://arxiv.org/abs/2206.00927 for the detailed derivation.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the sample tensor at the previous timestep.
        """
    lambda_t, lambda_s = self.lambda_t[prev_timestep], self.lambda_t[timestep]
    alpha_t, alpha_s = self.alpha_t[prev_timestep], self.alpha_t[timestep]
    sigma_t, sigma_s = self.sigma_t[prev_timestep], self.sigma_t[timestep]
    h = lambda_t - lambda_s
    if self.config.algorithm_type == 'dpmsolver++':
        x_t = sigma_t / sigma_s * sample - alpha_t * (torch.exp(-h) - 1.0
            ) * model_output
    elif self.config.algorithm_type == 'dpmsolver':
        x_t = alpha_t / alpha_s * sample - sigma_t * (torch.exp(h) - 1.0
            ) * model_output
    elif self.config.algorithm_type == 'sde-dpmsolver++':
        assert noise is not None
        x_t = sigma_t / sigma_s * torch.exp(-h) * sample + alpha_t * (1 -
            torch.exp(-2.0 * h)) * model_output + sigma_t * torch.sqrt(1.0 -
            torch.exp(-2 * h)) * noise
    elif self.config.algorithm_type == 'sde-dpmsolver':
        assert noise is not None
        x_t = alpha_t / alpha_s * sample - 2.0 * (sigma_t * (torch.exp(h) -
            1.0)) * model_output + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0
            ) * noise
    return x_t
