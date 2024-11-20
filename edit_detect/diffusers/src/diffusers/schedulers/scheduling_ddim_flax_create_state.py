def create_state(self, common: Optional[CommonSchedulerState]=None
    ) ->DDIMSchedulerState:
    if common is None:
        common = CommonSchedulerState.create(self)
    final_alpha_cumprod = jnp.array(1.0, dtype=self.dtype
        ) if self.config.set_alpha_to_one else common.alphas_cumprod[0]
    init_noise_sigma = jnp.array(1.0, dtype=self.dtype)
    timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]
    return DDIMSchedulerState.create(common=common, final_alpha_cumprod=
        final_alpha_cumprod, init_noise_sigma=init_noise_sigma, timesteps=
        timesteps)
