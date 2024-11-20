def step(self, state: DPMSolverMultistepSchedulerState, model_output: jnp.
    ndarray, timestep: int, sample: jnp.ndarray, return_dict: bool=True
    ) ->Union[FlaxDPMSolverMultistepSchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep by DPM-Solver. Core function to propagate the diffusion process
        from the learned model outputs (most often the predicted noise).

        Args:
            state (`DPMSolverMultistepSchedulerState`):
                the `FlaxDPMSolverMultistepScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than FlaxDPMSolverMultistepSchedulerOutput class

        Returns:
            [`FlaxDPMSolverMultistepSchedulerOutput`] or `tuple`: [`FlaxDPMSolverMultistepSchedulerOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
    if state.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
    step_index, = jnp.where(state.timesteps == timestep, size=1)
    step_index = step_index[0]
    prev_timestep = jax.lax.select(step_index == len(state.timesteps) - 1, 
        0, state.timesteps[step_index + 1])
    model_output = self.convert_model_output(state, model_output, timestep,
        sample)
    model_outputs_new = jnp.roll(state.model_outputs, -1, axis=0)
    model_outputs_new = model_outputs_new.at[-1].set(model_output)
    state = state.replace(model_outputs=model_outputs_new, prev_timestep=
        prev_timestep, cur_sample=sample)

    def step_1(state: DPMSolverMultistepSchedulerState) ->jnp.ndarray:
        return self.dpm_solver_first_order_update(state, state.
            model_outputs[-1], state.timesteps[step_index], state.
            prev_timestep, state.cur_sample)

    def step_23(state: DPMSolverMultistepSchedulerState) ->jnp.ndarray:

        def step_2(state: DPMSolverMultistepSchedulerState) ->jnp.ndarray:
            timestep_list = jnp.array([state.timesteps[step_index - 1],
                state.timesteps[step_index]])
            return self.multistep_dpm_solver_second_order_update(state,
                state.model_outputs, timestep_list, state.prev_timestep,
                state.cur_sample)

        def step_3(state: DPMSolverMultistepSchedulerState) ->jnp.ndarray:
            timestep_list = jnp.array([state.timesteps[step_index - 2],
                state.timesteps[step_index - 1], state.timesteps[step_index]])
            return self.multistep_dpm_solver_third_order_update(state,
                state.model_outputs, timestep_list, state.prev_timestep,
                state.cur_sample)
        step_2_output = step_2(state)
        step_3_output = step_3(state)
        if self.config.solver_order == 2:
            return step_2_output
        elif self.config.lower_order_final and len(state.timesteps) < 15:
            return jax.lax.select(state.lower_order_nums < 2, step_2_output,
                jax.lax.select(step_index == len(state.timesteps) - 2,
                step_2_output, step_3_output))
        else:
            return jax.lax.select(state.lower_order_nums < 2, step_2_output,
                step_3_output)
    step_1_output = step_1(state)
    step_23_output = step_23(state)
    if self.config.solver_order == 1:
        prev_sample = step_1_output
    elif self.config.lower_order_final and len(state.timesteps) < 15:
        prev_sample = jax.lax.select(state.lower_order_nums < 1,
            step_1_output, jax.lax.select(step_index == len(state.timesteps
            ) - 1, step_1_output, step_23_output))
    else:
        prev_sample = jax.lax.select(state.lower_order_nums < 1,
            step_1_output, step_23_output)
    state = state.replace(lower_order_nums=jnp.minimum(state.
        lower_order_nums + 1, self.config.solver_order))
    if not return_dict:
        return prev_sample, state
    return FlaxDPMSolverMultistepSchedulerOutput(prev_sample=prev_sample,
        state=state)
