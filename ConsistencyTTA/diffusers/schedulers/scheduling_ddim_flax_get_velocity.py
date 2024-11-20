def get_velocity(self, state: DDIMSchedulerState, sample: jnp.ndarray,
    noise: jnp.ndarray, timesteps: jnp.ndarray) ->jnp.ndarray:
    return get_velocity_common(state.common, sample, noise, timesteps)
