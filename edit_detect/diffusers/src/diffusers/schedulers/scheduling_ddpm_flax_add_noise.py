def add_noise(self, state: DDPMSchedulerState, original_samples: jnp.
    ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray) ->jnp.ndarray:
    return add_noise_common(state.common, original_samples, noise, timesteps)
