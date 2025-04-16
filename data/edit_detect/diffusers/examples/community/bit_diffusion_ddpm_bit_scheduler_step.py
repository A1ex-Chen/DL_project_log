def ddpm_bit_scheduler_step(self, model_output: torch.Tensor, timestep: int,
    sample: torch.Tensor, prediction_type='epsilon', generator=None,
    return_dict: bool=True) ->Union[DDPMSchedulerOutput, Tuple]:
    """
    Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
    process from the learned model outputs (most often the predicted noise).
    Args:
        model_output (`torch.Tensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.Tensor`):
            current instance of sample being created by diffusion process.
        prediction_type (`str`, default `epsilon`):
            indicates whether the model predicts the noise (epsilon), or the samples (`sample`).
        generator: random number generator.
        return_dict (`bool`): option for returning tuple rather than DDPMSchedulerOutput class
    Returns:
        [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] or `tuple`:
        [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
        returning a tuple, the first element is the sample tensor.
    """
    t = timestep
    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in [
        'learned', 'learned_range']:
        model_output, predicted_variance = torch.split(model_output, sample
            .shape[1], dim=1)
    else:
        predicted_variance = None
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else self.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    if prediction_type == 'epsilon':
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output
            ) / alpha_prod_t ** 0.5
    elif prediction_type == 'sample':
        pred_original_sample = model_output
    else:
        raise ValueError(f'Unsupported prediction_type {prediction_type}.')
    scale = self.bit_scale
    if self.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -scale, scale)
    pred_original_sample_coeff = alpha_prod_t_prev ** 0.5 * self.betas[t
        ] / beta_prod_t
    current_sample_coeff = self.alphas[t
        ] ** 0.5 * beta_prod_t_prev / beta_prod_t
    pred_prev_sample = (pred_original_sample_coeff * pred_original_sample +
        current_sample_coeff * sample)
    variance = 0
    if t > 0:
        noise = torch.randn(model_output.size(), dtype=model_output.dtype,
            layout=model_output.layout, generator=generator).to(model_output
            .device)
        variance = self._get_variance(t, predicted_variance=predicted_variance
            ) ** 0.5 * noise
    pred_prev_sample = pred_prev_sample + variance
    if not return_dict:
        return pred_prev_sample,
    return DDPMSchedulerOutput(prev_sample=pred_prev_sample,
        pred_original_sample=pred_original_sample)
