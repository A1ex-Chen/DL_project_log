def create_state(self, common: Optional[CommonSchedulerState]=None
    ) ->EulerDiscreteSchedulerState:
    if common is None:
        common = CommonSchedulerState.create(self)
    timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]
    sigmas = ((1 - common.alphas_cumprod) / common.alphas_cumprod) ** 0.5
    sigmas = jnp.interp(timesteps, jnp.arange(0, len(sigmas)), sigmas)
    sigmas = jnp.concatenate([sigmas, jnp.array([0.0], dtype=self.dtype)])
    if self.config.timestep_spacing in ['linspace', 'trailing']:
        init_noise_sigma = sigmas.max()
    else:
        init_noise_sigma = (sigmas.max() ** 2 + 1) ** 0.5
    return EulerDiscreteSchedulerState.create(common=common,
        init_noise_sigma=init_noise_sigma, timesteps=timesteps, sigmas=sigmas)
