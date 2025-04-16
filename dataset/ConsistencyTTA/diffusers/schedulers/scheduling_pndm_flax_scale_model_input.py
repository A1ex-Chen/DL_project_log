def scale_model_input(self, state: PNDMSchedulerState, sample: jnp.ndarray,
    timestep: Optional[int]=None) ->jnp.ndarray:
    """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            state (`PNDMSchedulerState`): the `FlaxPNDMScheduler` state data class instance.
            sample (`jnp.ndarray`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `jnp.ndarray`: scaled input sample
        """
    return sample
