def step(self, state: DDPMSchedulerState, model_output: jnp.ndarray,
    timestep: int, sample: jnp.ndarray, key: Optional[jax.random.KeyArray]=
    None, return_dict: bool=True) ->Union[FlaxDDPMSchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            state (`DDPMSchedulerState`): the `FlaxDDPMScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            key (`jax.random.KeyArray`): a PRNG key.
            return_dict (`bool`): option for returning tuple rather than FlaxDDPMSchedulerOutput class

        Returns:
            [`FlaxDDPMSchedulerOutput`] or `tuple`: [`FlaxDDPMSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.

        """
    t = timestep
    if key is None:
        key = jax.random.PRNGKey(0)
    if model_output.shape[1] == sample.shape[1
        ] * 2 and self.config.variance_type in ['learned', 'learned_range']:
        model_output, predicted_variance = jnp.split(model_output, sample.
            shape[1], axis=1)
    else:
        predicted_variance = None
    alpha_prod_t = state.common.alphas_cumprod[t]
    alpha_prod_t_prev = jnp.where(t > 0, state.common.alphas_cumprod[t - 1],
        jnp.array(1.0, dtype=self.dtype))
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    if self.config.prediction_type == 'epsilon':
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output
            ) / alpha_prod_t ** 0.5
    elif self.config.prediction_type == 'sample':
        pred_original_sample = model_output
    elif self.config.prediction_type == 'v_prediction':
        pred_original_sample = (alpha_prod_t ** 0.5 * sample - beta_prod_t **
            0.5 * model_output)
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`  for the FlaxDDPMScheduler.'
            )
    if self.config.clip_sample:
        pred_original_sample = jnp.clip(pred_original_sample, -1, 1)
    pred_original_sample_coeff = alpha_prod_t_prev ** 0.5 * state.common.betas[
        t] / beta_prod_t
    current_sample_coeff = state.common.alphas[t
        ] ** 0.5 * beta_prod_t_prev / beta_prod_t
    pred_prev_sample = (pred_original_sample_coeff * pred_original_sample +
        current_sample_coeff * sample)

    def random_variance():
        split_key = jax.random.split(key, num=1)
        noise = jax.random.normal(split_key, shape=model_output.shape,
            dtype=self.dtype)
        return self._get_variance(state, t, predicted_variance=
            predicted_variance) ** 0.5 * noise
    variance = jnp.where(t > 0, random_variance(), jnp.zeros(model_output.
        shape, dtype=self.dtype))
    pred_prev_sample = pred_prev_sample + variance
    if not return_dict:
        return pred_prev_sample, state
    return FlaxDDPMSchedulerOutput(prev_sample=pred_prev_sample, state=state)
