def step(self, state: EulerDiscreteSchedulerState, model_output: jnp.
    ndarray, timestep: int, sample: jnp.ndarray, return_dict: bool=True
    ) ->Union[FlaxEulerDiscreteSchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            state (`EulerDiscreteSchedulerState`):
                the `FlaxEulerDiscreteScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            order: coefficient for multi-step inference.
            return_dict (`bool`): option for returning tuple rather than FlaxEulerDiscreteScheduler class

        Returns:
            [`FlaxEulerDiscreteScheduler`] or `tuple`: [`FlaxEulerDiscreteScheduler`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
    if state.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
    step_index, = jnp.where(state.timesteps == timestep, size=1)
    step_index = step_index[0]
    sigma = state.sigmas[step_index]
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
    dt = state.sigmas[step_index + 1] - sigma
    prev_sample = sample + derivative * dt
    if not return_dict:
        return prev_sample, state
    return FlaxEulerDiscreteSchedulerOutput(prev_sample=prev_sample, state=
        state)
