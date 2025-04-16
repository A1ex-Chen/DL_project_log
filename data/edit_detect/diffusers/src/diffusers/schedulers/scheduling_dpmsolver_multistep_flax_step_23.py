def step_23(state: DPMSolverMultistepSchedulerState) ->jnp.ndarray:

    def step_2(state: DPMSolverMultistepSchedulerState) ->jnp.ndarray:
        timestep_list = jnp.array([state.timesteps[step_index - 1], state.
            timesteps[step_index]])
        return self.multistep_dpm_solver_second_order_update(state, state.
            model_outputs, timestep_list, state.prev_timestep, state.cur_sample
            )

    def step_3(state: DPMSolverMultistepSchedulerState) ->jnp.ndarray:
        timestep_list = jnp.array([state.timesteps[step_index - 2], state.
            timesteps[step_index - 1], state.timesteps[step_index]])
        return self.multistep_dpm_solver_third_order_update(state, state.
            model_outputs, timestep_list, state.prev_timestep, state.cur_sample
            )
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
