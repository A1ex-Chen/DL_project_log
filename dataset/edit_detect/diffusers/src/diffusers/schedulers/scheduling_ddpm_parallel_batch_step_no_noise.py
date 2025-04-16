def batch_step_no_noise(self, model_output: torch.Tensor, timesteps: List[
    int], sample: torch.Tensor) ->torch.Tensor:
    """
        Batched version of the `step` function, to be able to reverse the SDE for multiple samples/timesteps at once.
        Also, does not add any noise to the predicted sample, which is necessary for parallel sampling where the noise
        is pre-sampled by the pipeline.

        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`): direct output from learned diffusion model.
            timesteps (`List[int]`):
                current discrete timesteps in the diffusion chain. This is now a list of integers.
            sample (`torch.Tensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.Tensor`: sample tensor at previous timestep.
        """
    t = timesteps
    num_inference_steps = (self.num_inference_steps if self.
        num_inference_steps else self.config.num_train_timesteps)
    prev_t = t - self.config.num_train_timesteps // num_inference_steps
    t = t.view(-1, *([1] * (model_output.ndim - 1)))
    prev_t = prev_t.view(-1, *([1] * (model_output.ndim - 1)))
    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in [
        'learned', 'learned_range']:
        model_output, predicted_variance = torch.split(model_output, sample
            .shape[1], dim=1)
    else:
        pass
    self.alphas_cumprod = self.alphas_cumprod.to(model_output.device)
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[torch.clip(prev_t, min=0)]
    alpha_prod_t_prev[prev_t < 0] = torch.tensor(1.0)
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
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or `v_prediction`  for the DDPMParallelScheduler.'
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
    return pred_prev_sample
