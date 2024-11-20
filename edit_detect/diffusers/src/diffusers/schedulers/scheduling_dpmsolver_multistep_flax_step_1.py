def step_1(state: DPMSolverMultistepSchedulerState) ->jnp.ndarray:
    return self.dpm_solver_first_order_update(state, state.model_outputs[-1
        ], state.timesteps[step_index], state.prev_timestep, state.cur_sample)
