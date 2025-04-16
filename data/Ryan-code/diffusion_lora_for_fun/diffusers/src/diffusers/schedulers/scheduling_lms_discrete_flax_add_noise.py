def add_noise(self, state: LMSDiscreteSchedulerState, original_samples: jnp
    .ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray) ->jnp.ndarray:
    sigma = state.sigmas[timesteps].flatten()
    sigma = broadcast_to_shape_from_left(sigma, noise.shape)
    noisy_samples = original_samples + noise * sigma
    return noisy_samples
