def step_prk(self, state: PNDMSchedulerState, model_output: jnp.ndarray,
    timestep: int, sample: jnp.ndarray) ->Union[FlaxPNDMSchedulerOutput, Tuple
    ]:
    """
        Step function propagating the sample with the Runge-Kutta method. RK takes 4 forward passes to approximate the
        solution to the differential equation.

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
    diff_to_prev = jnp.where(state.counter % 2, 0, self.config.
        num_train_timesteps // state.num_inference_steps // 2)
    prev_timestep = timestep - diff_to_prev
    timestep = state.prk_timesteps[state.counter // 4 * 4]
    model_output = jax.lax.select(state.counter % 4 != 3, model_output, 
        state.cur_model_output + 1 / 6 * model_output)
    state = state.replace(cur_model_output=jax.lax.select_n(state.counter %
        4, state.cur_model_output + 1 / 6 * model_output, state.
        cur_model_output + 1 / 3 * model_output, state.cur_model_output + 1 /
        3 * model_output, jnp.zeros_like(state.cur_model_output)), ets=jax.
        lax.select(state.counter % 4 == 0, state.ets.at[0:3].set(state.ets[
        1:4]).at[3].set(model_output), state.ets), cur_sample=jax.lax.
        select(state.counter % 4 == 0, sample, state.cur_sample))
    cur_sample = state.cur_sample
    prev_sample = self._get_prev_sample(state, cur_sample, timestep,
        prev_timestep, model_output)
    state = state.replace(counter=state.counter + 1)
    return prev_sample, state
