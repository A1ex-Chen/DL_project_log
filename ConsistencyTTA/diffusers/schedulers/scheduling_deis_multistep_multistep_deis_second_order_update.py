def multistep_deis_second_order_update(self, model_output_list: List[torch.
    FloatTensor], timestep_list: List[int], prev_timestep: int, sample:
    torch.FloatTensor) ->torch.FloatTensor:
    """
        One step for the second-order multistep DEIS.

        Args:
            model_output_list (`List[torch.FloatTensor]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the sample tensor at the previous timestep.
        """
    t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
    m0, m1 = model_output_list[-1], model_output_list[-2]
    alpha_t, alpha_s0, alpha_s1 = self.alpha_t[t], self.alpha_t[s0
        ], self.alpha_t[s1]
    sigma_t, sigma_s0, sigma_s1 = self.sigma_t[t], self.sigma_t[s0
        ], self.sigma_t[s1]
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
