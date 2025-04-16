def _get_variance(self, state: DDPMSchedulerState, t, predicted_variance=
    None, variance_type=None):
    alpha_prod_t = state.common.alphas_cumprod[t]
    alpha_prod_t_prev = jnp.where(t > 0, state.common.alphas_cumprod[t - 1],
        jnp.array(1.0, dtype=self.dtype))
    variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t
        ) * state.common.betas[t]
    if variance_type is None:
        variance_type = self.config.variance_type
    if variance_type == 'fixed_small':
        variance = jnp.clip(variance, a_min=1e-20)
    elif variance_type == 'fixed_small_log':
        variance = jnp.log(jnp.clip(variance, a_min=1e-20))
    elif variance_type == 'fixed_large':
        variance = state.common.betas[t]
    elif variance_type == 'fixed_large_log':
        variance = jnp.log(state.common.betas[t])
    elif variance_type == 'learned':
        return predicted_variance
    elif variance_type == 'learned_range':
        min_log = variance
        max_log = state.common.betas[t]
        frac = (predicted_variance + 1) / 2
        variance = frac * max_log + (1 - frac) * min_log
    return variance
