@classmethod
def create(cls, common: CommonSchedulerState, alpha_t: jnp.ndarray, sigma_t:
    jnp.ndarray, lambda_t: jnp.ndarray, init_noise_sigma: jnp.ndarray,
    timesteps: jnp.ndarray):
    return cls(common=common, alpha_t=alpha_t, sigma_t=sigma_t, lambda_t=
        lambda_t, init_noise_sigma=init_noise_sigma, timesteps=timesteps)
