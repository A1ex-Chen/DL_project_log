def step(self, model_output: torch.FloatTensor, timestep: Union[int, torch.
    IntTensor], sample: torch.FloatTensor, generator=None, return_dict:
    bool=True) ->Union[DDPMSchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. 
        Core function to propagate the diffusion process from the learned model outputs 
        (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than DDPMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDPMSchedulerOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
    if not torch.is_tensor(timestep):
        timestep = torch.tensor(timestep)
    timestep = timestep.reshape(-1).to(sample.device)
    num_inference_steps = (self.num_inference_steps if self.
        num_inference_steps else self.config.num_train_timesteps)
    prev_t = timestep - self.config.num_train_timesteps // num_inference_steps
    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in [
        'learned', 'learned_range']:
        model_output, predicted_variance = torch.split(model_output, sample
            .shape[1], dim=1)
    else:
        predicted_variance = None
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = torch.where(prev_t >= 0, self.alphas_cumprod[prev_t
        ], self.one)
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
        pred_original_sample = (alpha_prod_t ** 0.5).reshape(-1, 1, 1, 1
            ) * sample - (beta_prod_t ** 0.5).reshape(-1, 1, 1, 1
            ) * model_output
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or `v_prediction` for the DDPMScheduler.'
            )
    if self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(-self.config.
            clip_sample_range, self.config.clip_sample_range)
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t /
        beta_prod_t)
    current_sample_coeff = (current_alpha_t ** 0.5 * beta_prod_t_prev /
        beta_prod_t)
    pred_prev_sample = pred_original_sample_coeff.reshape(-1, 1, 1, 1
        ) * pred_original_sample + current_sample_coeff.reshape(-1, 1, 1, 1
        ) * sample
    ind_to_get_var = (timestep > 0).nonzero().reshape(-1)
    model_output_ind = model_output[ind_to_get_var]
    if predicted_variance is None:
        predicted_variance_ind = None
    else:
        predicted_variance_ind = predicted_variance[ind_to_get_var]
    t_ind = timestep[ind_to_get_var]
    device = model_output.device
    variance_noise = randn_tensor(model_output_ind.shape, generator=
        generator, device=device, dtype=model_output_ind.dtype)
    if self.variance_type == 'fixed_small_log':
        variance_ind = self._get_variance(t_ind, predicted_variance=
            predicted_variance_ind) * variance_noise
    elif self.variance_type == 'learned_range':
        variance_ind = self._get_variance(t_ind, predicted_variance=
            predicted_variance_ind)
        variance_ind = torch.exp(0.5 * variance_ind) * variance_noise
    else:
        variance_ind = (self._get_variance(t_ind, predicted_variance=
            predicted_variance_ind) ** 0.5).reshape(-1, 1, 1, 1
            ) * variance_noise
    variance = torch.zeros_like(pred_prev_sample)
    variance[ind_to_get_var] = variance_ind
    pred_prev_sample = pred_prev_sample + variance
    if not return_dict:
        return pred_prev_sample,
    return DDPMSchedulerOutput(prev_sample=pred_prev_sample,
        pred_original_sample=pred_original_sample)
