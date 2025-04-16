def create_state(self, common: Optional[CommonSchedulerState]=None
    ) ->LMSDiscreteSchedulerState:
    if common is None:
        common = CommonSchedulerState.create(self)
    timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]
    sigmas = ((1 - common.alphas_cumprod) / common.alphas_cumprod) ** 0.5
    init_noise_sigma = sigmas.max()
    return LMSDiscreteSchedulerState.create(common=common, init_noise_sigma
        =init_noise_sigma, timesteps=timesteps, sigmas=sigmas)
