def multistep_dpm_solver_second_order_update(self, state:
    DPMSolverMultistepSchedulerState, model_output_list: jnp.ndarray,
    timestep_list: List[int], prev_timestep: int, sample: jnp.ndarray
    ) ->jnp.ndarray:
    """
        One step for the second-order multistep DPM-Solver.

        Args:
            model_output_list (`List[jnp.ndarray]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.

        Returns:
            `jnp.ndarray`: the sample tensor at the previous timestep.
        """
    t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
    m0, m1 = model_output_list[-1], model_output_list[-2]
    lambda_t, lambda_s0, lambda_s1 = state.lambda_t[t], state.lambda_t[s0
        ], state.lambda_t[s1]
    alpha_t, alpha_s0 = state.alpha_t[t], state.alpha_t[s0]
    sigma_t, sigma_s0 = state.sigma_t[t], state.sigma_t[s0]
    h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
    r0 = h_0 / h
    D0, D1 = m0, 1.0 / r0 * (m0 - m1)
    if self.config.algorithm_type == 'dpmsolver++':
        if self.config.solver_type == 'midpoint':
            x_t = sigma_t / sigma_s0 * sample - alpha_t * (jnp.exp(-h) - 1.0
                ) * D0 - 0.5 * (alpha_t * (jnp.exp(-h) - 1.0)) * D1
        elif self.config.solver_type == 'heun':
            x_t = sigma_t / sigma_s0 * sample - alpha_t * (jnp.exp(-h) - 1.0
                ) * D0 + alpha_t * ((jnp.exp(-h) - 1.0) / h + 1.0) * D1
    elif self.config.algorithm_type == 'dpmsolver':
        if self.config.solver_type == 'midpoint':
            x_t = alpha_t / alpha_s0 * sample - sigma_t * (jnp.exp(h) - 1.0
                ) * D0 - 0.5 * (sigma_t * (jnp.exp(h) - 1.0)) * D1
        elif self.config.solver_type == 'heun':
            x_t = alpha_t / alpha_s0 * sample - sigma_t * (jnp.exp(h) - 1.0
                ) * D0 - sigma_t * ((jnp.exp(h) - 1.0) / h - 1.0) * D1
    return x_t
