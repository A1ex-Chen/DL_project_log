def step_correct(self, state: ScoreSdeVeSchedulerState, model_output: jnp.
    ndarray, sample: jnp.ndarray, key: random.KeyArray, return_dict: bool=True
    ) ->Union[FlaxSdeVeOutput, Tuple]:
    """
        Correct the predicted sample based on the output model_output of the network. This is often run repeatedly
        after making the prediction for the previous timestep.

        Args:
            state (`ScoreSdeVeSchedulerState`): the `FlaxScoreSdeVeScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than FlaxSdeVeOutput class

        Returns:
            [`FlaxSdeVeOutput`] or `tuple`: [`FlaxSdeVeOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
    if state.timesteps is None:
        raise ValueError(
            "`state.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )
    key = random.split(key, num=1)
    noise = random.normal(key=key, shape=sample.shape)
    grad_norm = jnp.linalg.norm(model_output)
    noise_norm = jnp.linalg.norm(noise)
    step_size = (self.config.snr * noise_norm / grad_norm) ** 2 * 2
    step_size = step_size * jnp.ones(sample.shape[0])
    step_size = step_size.flatten()
    step_size = broadcast_to_shape_from_left(step_size, sample.shape)
    prev_sample_mean = sample + step_size * model_output
    prev_sample = prev_sample_mean + (step_size * 2) ** 0.5 * noise
    if not return_dict:
        return prev_sample, state
    return FlaxSdeVeOutput(prev_sample=prev_sample, state=state)
