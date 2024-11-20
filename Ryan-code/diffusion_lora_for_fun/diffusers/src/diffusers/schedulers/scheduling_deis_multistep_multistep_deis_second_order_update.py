def multistep_deis_second_order_update(self, model_output_list: List[torch.
    Tensor], *args, sample: torch.Tensor=None, **kwargs) ->torch.Tensor:
    """
        One step for the second-order multistep DEIS.

        Args:
            model_output_list (`List[torch.Tensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
    timestep_list = args[0] if len(args) > 0 else kwargs.pop('timestep_list',
        None)
    prev_timestep = args[1] if len(args) > 1 else kwargs.pop('prev_timestep',
        None)
    if sample is None:
        if len(args) > 2:
            sample = args[2]
        else:
            raise ValueError(' missing `sample` as a required keyward argument'
                )
    if timestep_list is not None:
        deprecate('timestep_list', '1.0.0',
            'Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`'
            )
    if prev_timestep is not None:
        deprecate('prev_timestep', '1.0.0',
            'Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`'
            )
    sigma_t, sigma_s0, sigma_s1 = self.sigmas[self.step_index + 1
        ], self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
    alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
    m0, m1 = model_output_list[-1], model_output_list[-2]
    rho_t, rho_s0, rho_s1 = (sigma_t / alpha_t, sigma_s0 / alpha_s0, 
        sigma_s1 / alpha_s1)
    if self.config.algorithm_type == 'deis':

        def ind_fn(t, b, c):
            return t * (-np.log(c) + np.log(t) - 1) / (np.log(b) - np.log(c))
        coef1 = ind_fn(rho_t, rho_s0, rho_s1) - ind_fn(rho_s0, rho_s0, rho_s1)
        coef2 = ind_fn(rho_t, rho_s1, rho_s0) - ind_fn(rho_s0, rho_s1, rho_s0)
        x_t = alpha_t * (sample / alpha_s0 + coef1 * m0 + coef2 * m1)
        return x_t
    else:
        raise NotImplementedError('only support log-rho multistep deis now')
