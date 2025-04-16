def step(self, model_output: torch.Tensor, sigma_hat: float, sigma_prev:
    float, sample_hat: torch.Tensor, return_dict: bool=True) ->Union[
    KarrasVeOutput, Tuple]:
    """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            sigma_hat (`float`):
            sigma_prev (`float`):
            sample_hat (`torch.Tensor`):
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_karras_ve.KarrasVESchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_karras_ve.KarrasVESchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_karras_ve.KarrasVESchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """
    pred_original_sample = sample_hat + sigma_hat * model_output
    derivative = (sample_hat - pred_original_sample) / sigma_hat
    sample_prev = sample_hat + (sigma_prev - sigma_hat) * derivative
    if not return_dict:
        return sample_prev, derivative
    return KarrasVeOutput(prev_sample=sample_prev, derivative=derivative,
        pred_original_sample=pred_original_sample)
