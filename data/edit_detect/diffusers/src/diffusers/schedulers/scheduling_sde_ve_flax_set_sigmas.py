def set_sigmas(self, state: ScoreSdeVeSchedulerState, num_inference_steps:
    int, sigma_min: float=None, sigma_max: float=None, sampling_eps: float=None
    ) ->ScoreSdeVeSchedulerState:
    """
        Sets the noise scales used for the diffusion chain. Supporting function to be run before inference.

        The sigmas control the weight of the `drift` and `diffusion` components of sample update.

        Args:
            state (`ScoreSdeVeSchedulerState`): the `FlaxScoreSdeVeScheduler` state data class instance.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            sigma_min (`float`, optional):
                initial noise scale value (overrides value given at Scheduler instantiation).
            sigma_max (`float`, optional):
                final noise scale value (overrides value given at Scheduler instantiation).
            sampling_eps (`float`, optional):
                final timestep value (overrides value given at Scheduler instantiation).
        """
    sigma_min = sigma_min if sigma_min is not None else self.config.sigma_min
    sigma_max = sigma_max if sigma_max is not None else self.config.sigma_max
    sampling_eps = (sampling_eps if sampling_eps is not None else self.
        config.sampling_eps)
    if state.timesteps is None:
        state = self.set_timesteps(state, num_inference_steps, sampling_eps)
    discrete_sigmas = jnp.exp(jnp.linspace(jnp.log(sigma_min), jnp.log(
        sigma_max), num_inference_steps))
    sigmas = jnp.array([(sigma_min * (sigma_max / sigma_min) ** t) for t in
        state.timesteps])
    return state.replace(discrete_sigmas=discrete_sigmas, sigmas=sigmas)
