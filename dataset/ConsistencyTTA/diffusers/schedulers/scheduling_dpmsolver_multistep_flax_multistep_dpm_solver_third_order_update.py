def multistep_dpm_solver_third_order_update(self, state:
    DPMSolverMultistepSchedulerState, model_output_list: jnp.ndarray,
    timestep_list: List[int], prev_timestep: int, sample: jnp.ndarray
    ) ->jnp.ndarray:
    """
        One step for the third-order multistep DPM-Solver.

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
    t, s0, s1, s2 = prev_timestep, timestep_list[-1], timestep_list[-2
        ], timestep_list[-3]
    m0, m1, m2 = model_output_list[-1], model_output_list[-2
        ], model_output_list[-3]
    lambda_t, lambda_s0, lambda_s1, lambda_s2 = state.lambda_t[t
        ], state.lambda_t[s0], state.lambda_t[s1], state.lambda_t[s2]
    alpha_t, alpha_s0 = state.alpha_t[t], state.alpha_t[s0]
    sigma_t, sigma_s0 = state.sigma_t[t], state.sigma_t[s0]
    h, h_0, h_1 = (lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 -
        lambda_s2)
    r0, r1 = h_0 / h, h_1 / h
    D0 = m0
    D1_0, D1_1 = 1.0 / r0 * (m0 - m1), 1.0 / r1 * (m1 - m2)
    D1 = D1_0 + r0 / (r0 + r1) * (D1_0 - D1_1)
    D2 = 1.0 / (r0 + r1) * (D1_0 - D1_1)
    if self.config.algorithm_type == 'dpmsolver++':
        x_t = sigma_t / sigma_s0 * sample - alpha_t * (jnp.exp(-h) - 1.0
            ) * D0 + alpha_t * ((jnp.exp(-h) - 1.0) / h + 1.0
            ) * D1 - alpha_t * ((jnp.exp(-h) - 1.0 + h) / h ** 2 - 0.5) * D2
    elif self.config.algorithm_type == 'dpmsolver':
        x_t = alpha_t / alpha_s0 * sample - sigma_t * (jnp.exp(h) - 1.0
            ) * D0 - sigma_t * ((jnp.exp(h) - 1.0) / h - 1.0
            ) * D1 - sigma_t * ((jnp.exp(h) - 1.0 - h) / h ** 2 - 0.5) * D2
    return x_t
