@classmethod
def create(cls, common: CommonSchedulerState, final_alpha_cumprod: jnp.
    ndarray, init_noise_sigma: jnp.ndarray, timesteps: jnp.ndarray):
    return cls(common=common, final_alpha_cumprod=final_alpha_cumprod,
        init_noise_sigma=init_noise_sigma, timesteps=timesteps)
