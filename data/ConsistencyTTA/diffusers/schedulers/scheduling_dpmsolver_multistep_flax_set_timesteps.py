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
    timesteps = jnp.linspace(0, self.config.num_train_timesteps - 1, 
        num_inference_steps + 1).round()[::-1][:-1].astype(jnp.int32)
    model_outputs = jnp.zeros((self.config.solver_order,) + shape, dtype=
        self.dtype)
    lower_order_nums = jnp.int32(0)
    prev_timestep = jnp.int32(-1)
    cur_sample = jnp.zeros(shape, dtype=self.dtype)
    return state.replace(num_inference_steps=num_inference_steps, timesteps
        =timesteps, model_outputs=model_outputs, lower_order_nums=
        lower_order_nums, prev_timestep=prev_timestep, cur_sample=cur_sample)
