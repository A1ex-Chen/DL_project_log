def step(self, model_output: torch.Tensor, timestep: int, sample: torch.
    Tensor, generator=None, return_dict: bool=True) ->Union[
    DDPMParallelSchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                current instance of sample being created by diffusion process.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than DDPMParallelSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDPMParallelSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDPMParallelSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.

        """
    t = timestep
    prev_t = self.previous_timestep(t)
    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in [
        'learned', 'learned_range']:
        model_output, predicted_variance = torch.split(model_output, sample
            .shape[1], dim=1)
    else:
        predicted_variance = None
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[prev_t
        ] if prev_t >= 0 else self.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t
    if self.config.prediction_type == 'epsilon':
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output
            ) / alpha_prod_t ** 0.5
    elif self.config.prediction_type == 'sample':
        pred_original_sample = model_output
    elif self.config.prediction_type == 'v_prediction':
        pred_original_sample = (alpha_prod_t ** 0.5 * sample - beta_prod_t **
            0.5 * model_output)
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or `v_prediction`  for the DDPMScheduler.'
            )
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(-self.config.
            clip_sample_range, self.config.clip_sample_range)
    pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t /
        beta_prod_t)
    current_sample_coeff = (current_alpha_t ** 0.5 * beta_prod_t_prev /
        beta_prod_t)
    pred_prev_sample = (pred_original_sample_coeff * pred_original_sample +
        current_sample_coeff * sample)
    variance = 0
    if t > 0:
        device = model_output.device
        variance_noise = randn_tensor(model_output.shape, generator=
            generator, device=device, dtype=model_output.dtype)
        if self.variance_type == 'fixed_small_log':
            variance = self._get_variance(t, predicted_variance=
                predicted_variance) * variance_noise
        elif self.variance_type == 'learned_range':
            variance = self._get_variance(t, predicted_variance=
                predicted_variance)
            variance = torch.exp(0.5 * variance) * variance_noise
        else:
            variance = self._get_variance(t, predicted_variance=
                predicted_variance) ** 0.5 * variance_noise
    pred_prev_sample = pred_prev_sample + variance
    if not return_dict:
        return pred_prev_sample,
    return DDPMParallelSchedulerOutput(prev_sample=pred_prev_sample,
        pred_original_sample=pred_original_sample)
