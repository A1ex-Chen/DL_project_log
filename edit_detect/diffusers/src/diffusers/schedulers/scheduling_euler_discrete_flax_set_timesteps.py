def set_timesteps(self, state: EulerDiscreteSchedulerState,
    num_inference_steps: int, shape: Tuple=()) ->EulerDiscreteSchedulerState:
    """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`EulerDiscreteSchedulerState`):
                the `FlaxEulerDiscreteScheduler` state data class instance.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
    if self.config.timestep_spacing == 'linspace':
        timesteps = jnp.linspace(self.config.num_train_timesteps - 1, 0,
            num_inference_steps, dtype=self.dtype)
    elif self.config.timestep_spacing == 'leading':
        step_ratio = self.config.num_train_timesteps // num_inference_steps
        timesteps = (jnp.arange(0, num_inference_steps) * step_ratio).round()[:
            :-1].copy().astype(float)
        timesteps += 1
    else:
        raise ValueError(
            f"timestep_spacing must be one of ['linspace', 'leading'], got {self.config.timestep_spacing}"
            )
    sigmas = ((1 - state.common.alphas_cumprod) / state.common.alphas_cumprod
        ) ** 0.5
    sigmas = jnp.interp(timesteps, jnp.arange(0, len(sigmas)), sigmas)
    sigmas = jnp.concatenate([sigmas, jnp.array([0.0], dtype=self.dtype)])
    if self.config.timestep_spacing in ['linspace', 'trailing']:
        init_noise_sigma = sigmas.max()
    else:
        init_noise_sigma = (sigmas.max() ** 2 + 1) ** 0.5
    return state.replace(timesteps=timesteps, sigmas=sigmas,
        num_inference_steps=num_inference_steps, init_noise_sigma=
        init_noise_sigma)
