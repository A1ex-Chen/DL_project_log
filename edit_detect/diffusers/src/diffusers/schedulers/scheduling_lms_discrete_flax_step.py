def step(self, state: LMSDiscreteSchedulerState, model_output: jnp.ndarray,
    timestep: int, sample: jnp.ndarray, order: int=4, return_dict: bool=True
    ) ->Union[FlaxLMSSchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            state (`LMSDiscreteSchedulerState`): the `FlaxLMSDiscreteScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            order: coefficient for multi-step inference.
            return_dict (`bool`): option for returning tuple rather than FlaxLMSSchedulerOutput class

        Returns:
            [`FlaxLMSSchedulerOutput`] or `tuple`: [`FlaxLMSSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.

        """
    if state.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
    sigma = state.sigmas[timestep]
    if self.config.prediction_type == 'epsilon':
        pred_original_sample = sample - sigma * model_output
    elif self.config.prediction_type == 'v_prediction':
        pred_original_sample = model_output * (-sigma / (sigma ** 2 + 1) ** 0.5
            ) + sample / (sigma ** 2 + 1)
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`'
            )
    derivative = (sample - pred_original_sample) / sigma
    state = state.replace(derivatives=jnp.append(state.derivatives, derivative)
        )
    if len(state.derivatives) > order:
        state = state.replace(derivatives=jnp.delete(state.derivatives, 0))
    order = min(timestep + 1, order)
    lms_coeffs = [self.get_lms_coefficient(state, order, timestep,
        curr_order) for curr_order in range(order)]
    prev_sample = sample + sum(coeff * derivative for coeff, derivative in
        zip(lms_coeffs, reversed(state.derivatives)))
    if not return_dict:
        return prev_sample, state
    return FlaxLMSSchedulerOutput(prev_sample=prev_sample, state=state)
