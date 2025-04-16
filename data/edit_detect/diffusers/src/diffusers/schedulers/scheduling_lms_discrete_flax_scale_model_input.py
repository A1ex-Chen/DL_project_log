def scale_model_input(self, state: LMSDiscreteSchedulerState, sample: jnp.
    ndarray, timestep: int) ->jnp.ndarray:
    """
        Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the K-LMS algorithm.

        Args:
            state (`LMSDiscreteSchedulerState`):
                the `FlaxLMSDiscreteScheduler` state data class instance.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            timestep (`int`):
                current discrete timestep in the diffusion chain.

        Returns:
            `jnp.ndarray`: scaled input sample
        """
    step_index, = jnp.where(state.timesteps == timestep, size=1)
    step_index = step_index[0]
    sigma = state.sigmas[step_index]
    sample = sample / (sigma ** 2 + 1) ** 0.5
    return sample
