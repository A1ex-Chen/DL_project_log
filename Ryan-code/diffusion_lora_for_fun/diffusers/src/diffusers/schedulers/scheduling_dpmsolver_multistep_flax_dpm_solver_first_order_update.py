def dpm_solver_first_order_update(self, state:
    DPMSolverMultistepSchedulerState, model_output: jnp.ndarray, timestep:
    int, prev_timestep: int, sample: jnp.ndarray) ->jnp.ndarray:
    """
        One step for the first-order DPM-Solver (equivalent to DDIM).

        See https://arxiv.org/abs/2206.00927 for the detailed derivation.

        Args:
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.

        Returns:
            `jnp.ndarray`: the sample tensor at the previous timestep.
        """
    t, s0 = prev_timestep, timestep
    m0 = model_output
    lambda_t, lambda_s = state.lambda_t[t], state.lambda_t[s0]
    alpha_t, alpha_s = state.alpha_t[t], state.alpha_t[s0]
    sigma_t, sigma_s = state.sigma_t[t], state.sigma_t[s0]
    h = lambda_t - lambda_s
    if self.config.algorithm_type == 'dpmsolver++':
        x_t = sigma_t / sigma_s * sample - alpha_t * (jnp.exp(-h) - 1.0) * m0
    elif self.config.algorithm_type == 'dpmsolver':
        x_t = alpha_t / alpha_s * sample - sigma_t * (jnp.exp(h) - 1.0) * m0
    return x_t
