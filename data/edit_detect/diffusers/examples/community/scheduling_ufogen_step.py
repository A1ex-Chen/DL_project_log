def step(self, model_output: torch.Tensor, timestep: int, sample: torch.
    Tensor, generator: Optional[torch.Generator]=None, return_dict: bool=True
    ) ->Union[UFOGenSchedulerOutput, Tuple]:
    """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ufogen.UFOGenSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.UFOGenSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ufogen.UFOGenSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
    t = timestep
    prev_t = self.previous_timestep(t)
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[prev_t
        ] if prev_t >= 0 else self.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
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
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or `v_prediction`  for UFOGenScheduler.'
            )
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(-self.config.
            clip_sample_range, self.config.clip_sample_range)
    if t != self.timesteps[-1]:
        device = model_output.device
        noise = randn_tensor(model_output.shape, generator=generator,
            device=device, dtype=model_output.dtype)
        sqrt_alpha_prod_t_prev = alpha_prod_t_prev ** 0.5
        sqrt_one_minus_alpha_prod_t_prev = (1 - alpha_prod_t_prev) ** 0.5
        pred_prev_sample = (sqrt_alpha_prod_t_prev * pred_original_sample +
            sqrt_one_minus_alpha_prod_t_prev * noise)
    else:
        pred_prev_sample = pred_original_sample
    if not return_dict:
        return pred_prev_sample,
    return UFOGenSchedulerOutput(prev_sample=pred_prev_sample,
        pred_original_sample=pred_original_sample)
