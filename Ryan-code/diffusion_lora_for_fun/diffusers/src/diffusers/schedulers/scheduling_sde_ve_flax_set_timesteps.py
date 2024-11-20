def set_timesteps(self, state: ScoreSdeVeSchedulerState,
    num_inference_steps: int, shape: Tuple=(), sampling_eps: float=None
    ) ->ScoreSdeVeSchedulerState:
    """
        Sets the continuous timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`ScoreSdeVeSchedulerState`): the `FlaxScoreSdeVeScheduler` state data class instance.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            sampling_eps (`float`, optional):
                final timestep value (overrides value given at Scheduler instantiation).

        """
    sampling_eps = (sampling_eps if sampling_eps is not None else self.
        config.sampling_eps)
    timesteps = jnp.linspace(1, sampling_eps, num_inference_steps)
    return state.replace(timesteps=timesteps)
