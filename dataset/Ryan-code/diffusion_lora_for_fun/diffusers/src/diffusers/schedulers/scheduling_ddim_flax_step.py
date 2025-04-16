def step(self, state: DDIMSchedulerState, model_output: jnp.ndarray,
    timestep: int, sample: jnp.ndarray, eta: float=0.0, return_dict: bool=True
    ) ->Union[FlaxDDIMSchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            state (`DDIMSchedulerState`): the `FlaxDDIMScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than FlaxDDIMSchedulerOutput class

        Returns:
            [`FlaxDDIMSchedulerOutput`] or `tuple`: [`FlaxDDIMSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.

        """
    if state.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
    prev_timestep = (timestep - self.config.num_train_timesteps // state.
        num_inference_steps)
    alphas_cumprod = state.common.alphas_cumprod
    final_alpha_cumprod = state.final_alpha_cumprod
    alpha_prod_t = alphas_cumprod[timestep]
    alpha_prod_t_prev = jnp.where(prev_timestep >= 0, alphas_cumprod[
        prev_timestep], final_alpha_cumprod)
    beta_prod_t = 1 - alpha_prod_t
    if self.config.prediction_type == 'epsilon':
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output
            ) / alpha_prod_t ** 0.5
        pred_epsilon = model_output
    elif self.config.prediction_type == 'sample':
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample
            ) / beta_prod_t ** 0.5
    elif self.config.prediction_type == 'v_prediction':
        pred_original_sample = (alpha_prod_t ** 0.5 * sample - beta_prod_t **
            0.5 * model_output)
        pred_epsilon = (alpha_prod_t ** 0.5 * model_output + beta_prod_t **
            0.5 * sample)
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`'
            )
    if self.config.clip_sample:
        pred_original_sample = pred_original_sample.clip(-self.config.
            clip_sample_range, self.config.clip_sample_range)
    variance = self._get_variance(state, timestep, prev_timestep)
    std_dev_t = eta * variance ** 0.5
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2
        ) ** 0.5 * pred_epsilon
    prev_sample = (alpha_prod_t_prev ** 0.5 * pred_original_sample +
        pred_sample_direction)
    if not return_dict:
        return prev_sample, state
    return FlaxDDIMSchedulerOutput(prev_sample=prev_sample, state=state)
