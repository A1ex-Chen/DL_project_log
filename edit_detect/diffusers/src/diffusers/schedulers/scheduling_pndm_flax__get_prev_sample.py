def _get_prev_sample(self, state: PNDMSchedulerState, sample, timestep,
    prev_timestep, model_output):
    alpha_prod_t = state.common.alphas_cumprod[timestep]
    alpha_prod_t_prev = jnp.where(prev_timestep >= 0, state.common.
        alphas_cumprod[prev_timestep], state.final_alpha_cumprod)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    if self.config.prediction_type == 'v_prediction':
        model_output = (alpha_prod_t ** 0.5 * model_output + beta_prod_t **
            0.5 * sample)
    elif self.config.prediction_type != 'epsilon':
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon` or `v_prediction`'
            )
    sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
    model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev ** 0.5 + (
        alpha_prod_t * beta_prod_t * alpha_prod_t_prev) ** 0.5
    prev_sample = sample_coeff * sample - (alpha_prod_t_prev - alpha_prod_t
        ) * model_output / model_output_denom_coeff
    return prev_sample
