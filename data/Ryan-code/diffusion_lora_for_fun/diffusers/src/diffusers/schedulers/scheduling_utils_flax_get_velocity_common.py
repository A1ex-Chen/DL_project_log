def get_velocity_common(state: CommonSchedulerState, sample: jnp.ndarray,
    noise: jnp.ndarray, timesteps: jnp.ndarray):
    sqrt_alpha_prod, sqrt_one_minus_alpha_prod = get_sqrt_alpha_prod(state,
        sample, noise, timesteps)
    velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
    return velocity
