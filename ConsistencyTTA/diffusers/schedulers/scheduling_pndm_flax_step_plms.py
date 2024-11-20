def step_plms(self, state: PNDMSchedulerState, model_output: jnp.ndarray,
    timestep: int, sample: jnp.ndarray) ->Union[FlaxPNDMSchedulerOutput, Tuple
    ]:
    """
        Step function propagating the sample with the linear multi-step method. This has one forward pass with multiple
        times to approximate the solution.

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
    prev_timestep = (timestep - self.config.num_train_timesteps // state.
        num_inference_steps)
    prev_timestep = jnp.where(prev_timestep > 0, prev_timestep, 0)
    prev_timestep = jnp.where(state.counter == 1, timestep, prev_timestep)
    timestep = jnp.where(state.counter == 1, timestep + self.config.
        num_train_timesteps // state.num_inference_steps, timestep)
    state = state.replace(ets=jax.lax.select(state.counter != 1, state.ets.
        at[0:3].set(state.ets[1:4]).at[3].set(model_output), state.ets),
        cur_sample=jax.lax.select(state.counter != 1, sample, state.cur_sample)
        )
    state = state.replace(cur_model_output=jax.lax.select_n(jnp.clip(state.
        counter, 0, 4), model_output, (model_output + state.ets[-1]) / 2, (
        3 * state.ets[-1] - state.ets[-2]) / 2, (23 * state.ets[-1] - 16 *
        state.ets[-2] + 5 * state.ets[-3]) / 12, 1 / 24 * (55 * state.ets[-
        1] - 59 * state.ets[-2] + 37 * state.ets[-3] - 9 * state.ets[-4])))
    sample = state.cur_sample
    model_output = state.cur_model_output
    prev_sample = self._get_prev_sample(state, sample, timestep,
        prev_timestep, model_output)
    state = state.replace(counter=state.counter + 1)
    return prev_sample, state
