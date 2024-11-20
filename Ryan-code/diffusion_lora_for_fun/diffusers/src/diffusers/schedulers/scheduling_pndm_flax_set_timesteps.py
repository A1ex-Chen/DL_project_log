def set_timesteps(self, state: PNDMSchedulerState, num_inference_steps: int,
    shape: Tuple) ->PNDMSchedulerState:
    """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`PNDMSchedulerState`):
                the `FlaxPNDMScheduler` state data class instance.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            shape (`Tuple`):
                the shape of the samples to be generated.
        """
    step_ratio = self.config.num_train_timesteps // num_inference_steps
    _timesteps = (jnp.arange(0, num_inference_steps) * step_ratio).round(
        ) + self.config.steps_offset
    if self.config.skip_prk_steps:
        prk_timesteps = jnp.array([], dtype=jnp.int32)
        plms_timesteps = jnp.concatenate([_timesteps[:-1], _timesteps[-2:-1
            ], _timesteps[-1:]])[::-1]
    else:
        prk_timesteps = _timesteps[-self.pndm_order:].repeat(2) + jnp.tile(jnp
            .array([0, self.config.num_train_timesteps //
            num_inference_steps // 2], dtype=jnp.int32), self.pndm_order)
        prk_timesteps = prk_timesteps[:-1].repeat(2)[1:-1][::-1]
        plms_timesteps = _timesteps[:-3][::-1]
    timesteps = jnp.concatenate([prk_timesteps, plms_timesteps])
    cur_model_output = jnp.zeros(shape, dtype=self.dtype)
    counter = jnp.int32(0)
    cur_sample = jnp.zeros(shape, dtype=self.dtype)
    ets = jnp.zeros((4,) + shape, dtype=self.dtype)
    return state.replace(timesteps=timesteps, num_inference_steps=
        num_inference_steps, prk_timesteps=prk_timesteps, plms_timesteps=
        plms_timesteps, cur_model_output=cur_model_output, counter=counter,
        cur_sample=cur_sample, ets=ets)
