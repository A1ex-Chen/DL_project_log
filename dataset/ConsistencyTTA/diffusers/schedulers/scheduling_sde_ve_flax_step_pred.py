def step_pred(self, state: ScoreSdeVeSchedulerState, model_output: jnp.
    ndarray, timestep: int, sample: jnp.ndarray, key: random.KeyArray,
    return_dict: bool=True) ->Union[FlaxSdeVeOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            state (`ScoreSdeVeSchedulerState`): the `FlaxScoreSdeVeScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
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
    timestep = timestep * jnp.ones(sample.shape[0])
    timesteps = (timestep * (len(state.timesteps) - 1)).long()
    sigma = state.discrete_sigmas[timesteps]
    adjacent_sigma = self.get_adjacent_sigma(state, timesteps, timestep)
    drift = jnp.zeros_like(sample)
    diffusion = (sigma ** 2 - adjacent_sigma ** 2) ** 0.5
    diffusion = diffusion.flatten()
    diffusion = broadcast_to_shape_from_left(diffusion, sample.shape)
    drift = drift - diffusion ** 2 * model_output
    key = random.split(key, num=1)
    noise = random.normal(key=key, shape=sample.shape)
    prev_sample_mean = sample - drift
    prev_sample = prev_sample_mean + diffusion * noise
    if not return_dict:
        return prev_sample, prev_sample_mean, state
    return FlaxSdeVeOutput(prev_sample=prev_sample, prev_sample_mean=
        prev_sample_mean, state=state)
