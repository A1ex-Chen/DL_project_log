def step_2(state: DPMSolverMultistepSchedulerState) ->jnp.ndarray:
    timestep_list = jnp.array([state.timesteps[step_index - 1], state.
        timesteps[step_index]])
    return self.multistep_dpm_solver_second_order_update(state, state.
        model_outputs, timestep_list, state.prev_timestep, state.cur_sample)
