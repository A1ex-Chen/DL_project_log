def step(self, model_output: torch.FloatTensor, timestep: int, sample:
    torch.FloatTensor, prev_timestep: Optional[int]=None, generator=None,
    return_dict: bool=True) ->Union[UnCLIPSchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            prev_timestep (`int`, *optional*): The previous timestep to predict the previous sample at.
                Used to dynamically compute beta. If not given, `t-1` is used and the pre-computed beta is used.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than UnCLIPSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.UnCLIPSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.UnCLIPSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
    t = timestep
    if model_output.shape[1] == sample.shape[1
        ] * 2 and self.variance_type == 'learned_range':
        model_output, predicted_variance = torch.split(model_output, sample
            .shape[1], dim=1)
    else:
        predicted_variance = None
    if prev_timestep is None:
        prev_timestep = t - 1
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep
        ] if prev_timestep >= 0 else self.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    if prev_timestep == t - 1:
        beta = self.betas[t]
        alpha = self.alphas[t]
    else:
        beta = 1 - alpha_prod_t / alpha_prod_t_prev
        alpha = 1 - beta
    if self.config.prediction_type == 'epsilon':
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output
            ) / alpha_prod_t ** 0.5
    elif self.config.prediction_type == 'sample':
        pred_original_sample = model_output
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon` or `sample` for the UnCLIPScheduler.'
            )
    if self.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -self.
            config.clip_sample_range, self.config.clip_sample_range)
    pred_original_sample_coeff = alpha_prod_t_prev ** 0.5 * beta / beta_prod_t
    current_sample_coeff = alpha ** 0.5 * beta_prod_t_prev / beta_prod_t
    pred_prev_sample = (pred_original_sample_coeff * pred_original_sample +
        current_sample_coeff * sample)
    variance = 0
    if t > 0:
        variance_noise = randn_tensor(model_output.shape, dtype=
            model_output.dtype, generator=generator, device=model_output.device
            )
        variance = self._get_variance(t, predicted_variance=
            predicted_variance, prev_timestep=prev_timestep)
        if self.variance_type == 'fixed_small_log':
            variance = variance
        elif self.variance_type == 'learned_range':
            variance = (0.5 * variance).exp()
        else:
            raise ValueError(
                f'variance_type given as {self.variance_type} must be one of `fixed_small_log` or `learned_range` for the UnCLIPScheduler.'
                )
        variance = variance * variance_noise
    pred_prev_sample = pred_prev_sample + variance
    if not return_dict:
        return pred_prev_sample,
    return UnCLIPSchedulerOutput(prev_sample=pred_prev_sample,
        pred_original_sample=pred_original_sample)
