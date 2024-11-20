def step(self, state: KarrasVeSchedulerState, model_output: jnp.ndarray,
    sigma_hat: float, sigma_prev: float, sample_hat: jnp.ndarray,
    return_dict: bool=True) ->Union[FlaxKarrasVeOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            state (`KarrasVeSchedulerState`): the `FlaxKarrasVeScheduler` state data class.
            model_output (`torch.Tensor` or `np.ndarray`): direct output from learned diffusion model.
            sigma_hat (`float`): TODO
            sigma_prev (`float`): TODO
            sample_hat (`torch.Tensor` or `np.ndarray`): TODO
            return_dict (`bool`): option for returning tuple rather than FlaxKarrasVeOutput class

        Returns:
            [`~schedulers.scheduling_karras_ve_flax.FlaxKarrasVeOutput`] or `tuple`: Updated sample in the diffusion
            chain and derivative. [`~schedulers.scheduling_karras_ve_flax.FlaxKarrasVeOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
    pred_original_sample = sample_hat + sigma_hat * model_output
    derivative = (sample_hat - pred_original_sample) / sigma_hat
    sample_prev = sample_hat + (sigma_prev - sigma_hat) * derivative
    if not return_dict:
        return sample_prev, derivative, state
    return FlaxKarrasVeOutput(prev_sample=sample_prev, derivative=
        derivative, state=state)
