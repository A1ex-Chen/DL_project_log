def _get_variance(self, state: DDIMSchedulerState, timestep, prev_timestep):
    alpha_prod_t = state.common.alphas_cumprod[timestep]
    alpha_prod_t_prev = jnp.where(prev_timestep >= 0, state.common.
        alphas_cumprod[prev_timestep], state.final_alpha_cumprod)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = beta_prod_t_prev / beta_prod_t * (1 - alpha_prod_t /
        alpha_prod_t_prev)
    return variance
