@classmethod
def create(cls, common: CommonSchedulerState, init_noise_sigma: jnp.ndarray,
    timesteps: jnp.ndarray):
    return cls(common=common, init_noise_sigma=init_noise_sigma, timesteps=
        timesteps)
