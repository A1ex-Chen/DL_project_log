def add_noise_common(state: CommonSchedulerState, original_samples: jnp.
    ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray):
    sqrt_alpha_prod, sqrt_one_minus_alpha_prod = get_sqrt_alpha_prod(state,
        original_samples, noise, timesteps)
    noisy_samples = (sqrt_alpha_prod * original_samples + 
        sqrt_one_minus_alpha_prod * noise)
    return noisy_samples
