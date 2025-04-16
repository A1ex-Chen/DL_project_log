def step(self, state: PNDMSchedulerState, model_output: jnp.ndarray,
    timestep: int, sample: jnp.ndarray, return_dict: bool=True) ->Union[
    FlaxPNDMSchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        This function calls `step_prk()` or `step_plms()` depending on the internal variable `counter`.

        Args:
            state (`PNDMSchedulerState`): the `FlaxPNDMScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than FlaxPNDMSchedulerOutput class

        Returns:
            [`FlaxPNDMSchedulerOutput`] or `tuple`: [`FlaxPNDMSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.

        """
    if state.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
    if self.config.skip_prk_steps:
        prev_sample, state = self.step_plms(state, model_output, timestep,
            sample)
    else:
        prk_prev_sample, prk_state = self.step_prk(state, model_output,
            timestep, sample)
        plms_prev_sample, plms_state = self.step_plms(state, model_output,
            timestep, sample)
        cond = state.counter < len(state.prk_timesteps)
        prev_sample = jax.lax.select(cond, prk_prev_sample, plms_prev_sample)
        state = state.replace(cur_model_output=jax.lax.select(cond,
            prk_state.cur_model_output, plms_state.cur_model_output), ets=
            jax.lax.select(cond, prk_state.ets, plms_state.ets), cur_sample
            =jax.lax.select(cond, prk_state.cur_sample, plms_state.
            cur_sample), counter=jax.lax.select(cond, prk_state.counter,
            plms_state.counter))
    if not return_dict:
        return prev_sample, state
    return FlaxPNDMSchedulerOutput(prev_sample=prev_sample, state=state)
