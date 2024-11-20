def create_state(self, common: Optional[CommonSchedulerState]=None
    ) ->DDPMSchedulerState:
    if common is None:
        common = CommonSchedulerState.create(self)
    init_noise_sigma = jnp.array(1.0, dtype=self.dtype)
    timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]
    return DDPMSchedulerState.create(common=common, init_noise_sigma=
        init_noise_sigma, timesteps=timesteps)
