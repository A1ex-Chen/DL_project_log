def set_timesteps(self, state: KarrasVeSchedulerState, num_inference_steps:
    int, shape: Tuple=()) ->KarrasVeSchedulerState:
    """
        Sets the continuous timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`KarrasVeSchedulerState`):
                the `FlaxKarrasVeScheduler` state data class.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.

        """
    timesteps = jnp.arange(0, num_inference_steps)[::-1].copy()
    schedule = [(self.config.sigma_max ** 2 * (self.config.sigma_min ** 2 /
        self.config.sigma_max ** 2) ** (i / (num_inference_steps - 1))) for
        i in timesteps]
    return state.replace(num_inference_steps=num_inference_steps, schedule=
        jnp.array(schedule, dtype=jnp.float32), timesteps=timesteps)
