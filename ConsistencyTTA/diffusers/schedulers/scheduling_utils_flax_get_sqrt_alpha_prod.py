def get_sqrt_alpha_prod(state: CommonSchedulerState, original_samples: jnp.
    ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray):
    alphas_cumprod = state.alphas_cumprod
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    sqrt_alpha_prod = broadcast_to_shape_from_left(sqrt_alpha_prod,
        original_samples.shape)
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    sqrt_one_minus_alpha_prod = broadcast_to_shape_from_left(
        sqrt_one_minus_alpha_prod, original_samples.shape)
    return sqrt_alpha_prod, sqrt_one_minus_alpha_prod
