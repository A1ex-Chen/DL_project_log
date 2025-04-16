def scale_model_input(self, state: DPMSolverMultistepSchedulerState, sample:
    jnp.ndarray, timestep: Optional[int]=None) ->jnp.ndarray:
    """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            state (`DPMSolverMultistepSchedulerState`):
                the `FlaxDPMSolverMultistepScheduler` state data class instance.
            sample (`jnp.ndarray`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `jnp.ndarray`: scaled input sample
        """
    return sample
