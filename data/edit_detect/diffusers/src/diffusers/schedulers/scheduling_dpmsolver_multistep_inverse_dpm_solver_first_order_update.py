def dpm_solver_first_order_update(self, model_output: torch.Tensor, *args,
    sample: torch.Tensor=None, noise: Optional[torch.Tensor]=None, **kwargs
    ) ->torch.Tensor:
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
    timestep = args[0] if len(args) > 0 else kwargs.pop('timestep', None)
    prev_timestep = args[1] if len(args) > 1 else kwargs.pop('prev_timestep',
        None)
    if sample is None:
        if len(args) > 2:
            sample = args[2]
        else:
            raise ValueError(' missing `sample` as a required keyward argument'
                )
    if timestep is not None:
        deprecate('timesteps', '1.0.0',
            'Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`'
            )
    if prev_timestep is not None:
        deprecate('prev_timestep', '1.0.0',
            'Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`'
            )
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
