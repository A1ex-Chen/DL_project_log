def batch_step_no_noise(self, model_output: torch.Tensor, timesteps: List[
    int], sample: torch.Tensor, eta: float=0.0, use_clipped_model_output:
    bool=False) ->torch.Tensor:
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
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
                predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
                `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
                coincide with the one provided as input and `use_clipped_model_output` will have not effect.

        Returns:
            `torch.Tensor`: sample tensor at previous timestep.

        """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
    assert eta == 0.0
    t = timesteps
    prev_t = t - self.config.num_train_timesteps // self.num_inference_steps
    t = t.view(-1, *([1] * (model_output.ndim - 1)))
    prev_t = prev_t.view(-1, *([1] * (model_output.ndim - 1)))
    self.alphas_cumprod = self.alphas_cumprod.to(model_output.device)
    self.final_alpha_cumprod = self.final_alpha_cumprod.to(model_output.device)
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[torch.clip(prev_t, min=0)]
    alpha_prod_t_prev[prev_t < 0] = torch.tensor(1.0)
    beta_prod_t = 1 - alpha_prod_t
    if self.config.prediction_type == 'epsilon':
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output
            ) / alpha_prod_t ** 0.5
        pred_epsilon = model_output
    elif self.config.prediction_type == 'sample':
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample
            ) / beta_prod_t ** 0.5
    elif self.config.prediction_type == 'v_prediction':
        pred_original_sample = (alpha_prod_t ** 0.5 * sample - beta_prod_t **
            0.5 * model_output)
        pred_epsilon = (alpha_prod_t ** 0.5 * model_output + beta_prod_t **
            0.5 * sample)
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`'
            )
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(-self.config.
            clip_sample_range, self.config.clip_sample_range)
    variance = self._batch_get_variance(t, prev_t).to(model_output.device
        ).view(*alpha_prod_t_prev.shape)
    std_dev_t = eta * variance ** 0.5
    if use_clipped_model_output:
        pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample
            ) / beta_prod_t ** 0.5
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2
        ) ** 0.5 * pred_epsilon
    prev_sample = (alpha_prod_t_prev ** 0.5 * pred_original_sample +
        pred_sample_direction)
    return prev_sample
