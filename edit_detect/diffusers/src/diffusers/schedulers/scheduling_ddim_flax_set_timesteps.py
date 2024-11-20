def set_timesteps(self, state: DDIMSchedulerState, num_inference_steps: int,
    shape: Tuple=()) ->DDIMSchedulerState:
    """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`DDIMSchedulerState`):
                the `FlaxDDIMScheduler` state data class instance.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
    step_ratio = self.config.num_train_timesteps // num_inference_steps
    timesteps = (jnp.arange(0, num_inference_steps) * step_ratio).round()[::-1
        ] + self.config.steps_offset
    return state.replace(num_inference_steps=num_inference_steps, timesteps
        =timesteps)
