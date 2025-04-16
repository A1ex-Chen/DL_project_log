def create_state(self, common: Optional[CommonSchedulerState]=None
    ) ->DPMSolverMultistepSchedulerState:
    if common is None:
        common = CommonSchedulerState.create(self)
    alpha_t = jnp.sqrt(common.alphas_cumprod)
    sigma_t = jnp.sqrt(1 - common.alphas_cumprod)
    lambda_t = jnp.log(alpha_t) - jnp.log(sigma_t)
    if self.config.algorithm_type not in ['dpmsolver', 'dpmsolver++']:
        raise NotImplementedError(
            f'{self.config.algorithm_type} does is not implemented for {self.__class__}'
            )
    if self.config.solver_type not in ['midpoint', 'heun']:
        raise NotImplementedError(
            f'{self.config.solver_type} does is not implemented for {self.__class__}'
            )
    init_noise_sigma = jnp.array(1.0, dtype=self.dtype)
    timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]
    return DPMSolverMultistepSchedulerState.create(common=common, alpha_t=
        alpha_t, sigma_t=sigma_t, lambda_t=lambda_t, init_noise_sigma=
        init_noise_sigma, timesteps=timesteps)
