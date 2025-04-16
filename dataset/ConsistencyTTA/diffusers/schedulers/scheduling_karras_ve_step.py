def step(self, model_output: torch.FloatTensor, sigma_hat: float,
    sigma_prev: float, sample_hat: torch.FloatTensor, return_dict: bool=True
    ) ->Union[KarrasVeOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            sigma_hat (`float`): TODO
            sigma_prev (`float`): TODO
            sample_hat (`torch.FloatTensor`): TODO
            return_dict (`bool`): option for returning tuple rather than KarrasVeOutput class

            KarrasVeOutput: updated sample in the diffusion chain and derivative (TODO double check).
        Returns:
            [`~schedulers.scheduling_karras_ve.KarrasVeOutput`] or `tuple`:
            [`~schedulers.scheduling_karras_ve.KarrasVeOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
    pred_original_sample = sample_hat + sigma_hat * model_output
    derivative = (sample_hat - pred_original_sample) / sigma_hat
    sample_prev = sample_hat + (sigma_prev - sigma_hat) * derivative
    if not return_dict:
        return sample_prev, derivative
    return KarrasVeOutput(prev_sample=sample_prev, derivative=derivative,
        pred_original_sample=pred_original_sample)
