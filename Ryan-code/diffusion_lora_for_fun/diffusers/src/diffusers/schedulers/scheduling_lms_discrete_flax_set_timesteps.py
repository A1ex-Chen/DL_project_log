def set_timesteps(self, state: LMSDiscreteSchedulerState,
    num_inference_steps: int, shape: Tuple=()) ->LMSDiscreteSchedulerState:
    """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`LMSDiscreteSchedulerState`):
                the `FlaxLMSDiscreteScheduler` state data class instance.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
    timesteps = jnp.linspace(self.config.num_train_timesteps - 1, 0,
        num_inference_steps, dtype=self.dtype)
    low_idx = jnp.floor(timesteps).astype(jnp.int32)
    high_idx = jnp.ceil(timesteps).astype(jnp.int32)
    frac = jnp.mod(timesteps, 1.0)
    sigmas = ((1 - state.common.alphas_cumprod) / state.common.alphas_cumprod
        ) ** 0.5
    sigmas = (1 - frac) * sigmas[low_idx] + frac * sigmas[high_idx]
    sigmas = jnp.concatenate([sigmas, jnp.array([0.0], dtype=self.dtype)])
    timesteps = timesteps.astype(jnp.int32)
    derivatives = jnp.zeros((0,) + shape, dtype=self.dtype)
    return state.replace(timesteps=timesteps, sigmas=sigmas,
        num_inference_steps=num_inference_steps, derivatives=derivatives)
