def set_timesteps(self, state: DPMSolverMultistepSchedulerState,
    num_inference_steps: int, shape: Tuple) ->DPMSolverMultistepSchedulerState:
    """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`DPMSolverMultistepSchedulerState`):
                the `FlaxDPMSolverMultistepScheduler` state data class instance.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            shape (`Tuple`):
                the shape of the samples to be generated.
        """
    last_timestep = self.config.num_train_timesteps
    if self.config.timestep_spacing == 'linspace':
        timesteps = jnp.linspace(0, last_timestep - 1, num_inference_steps + 1
            ).round()[::-1][:-1].astype(jnp.int32)
    elif self.config.timestep_spacing == 'leading':
        step_ratio = last_timestep // (num_inference_steps + 1)
        timesteps = (jnp.arange(0, num_inference_steps + 1) * step_ratio
            ).round()[::-1][:-1].copy().astype(jnp.int32)
        timesteps += self.config.steps_offset
    elif self.config.timestep_spacing == 'trailing':
        step_ratio = self.config.num_train_timesteps / num_inference_steps
        timesteps = jnp.arange(last_timestep, 0, -step_ratio).round().copy(
            ).astype(jnp.int32)
        timesteps -= 1
    else:
        raise ValueError(
            f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )
    model_outputs = jnp.zeros((self.config.solver_order,) + shape, dtype=
        self.dtype)
    lower_order_nums = jnp.int32(0)
    prev_timestep = jnp.int32(-1)
    cur_sample = jnp.zeros(shape, dtype=self.dtype)
    return state.replace(num_inference_steps=num_inference_steps, timesteps
        =timesteps, model_outputs=model_outputs, lower_order_nums=
        lower_order_nums, prev_timestep=prev_timestep, cur_sample=cur_sample)
