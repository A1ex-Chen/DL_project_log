def scale_model_input(self, state: DDPMSchedulerState, sample: jnp.ndarray,
    timestep: Optional[int]=None) ->jnp.ndarray:
    """
        Args:
            state (`PNDMSchedulerState`): the `FlaxPNDMScheduler` state data class instance.
            sample (`jnp.ndarray`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `jnp.ndarray`: scaled input sample
        """
    return sample
