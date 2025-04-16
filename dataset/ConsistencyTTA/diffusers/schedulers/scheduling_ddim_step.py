def step(self, model_output: torch.FloatTensor, timestep: Union[int, torch.
    IntTensor], sample: torch.FloatTensor, eta: float=0.0,
    use_clipped_model_output: bool=False, generator=None, variance_noise:
    Optional[torch.FloatTensor]=None, return_dict: bool=True) ->Union[
    DDIMSchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
                predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
                `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
                coincide with the one provided as input and `use_clipped_model_output` will have not effect.
            generator: random number generator.
            variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
                can directly provide the noise for the variance itself. This is useful for methods such as
                CycleDiffusion. (https://arxiv.org/abs/2210.05559)
            return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
    if not torch.is_tensor(timestep):
        timestep = torch.tensor(timestep)
    timestep = timestep.reshape(-1).to(sample.device)
    prev_timestep = timestep.cpu(
        ) - self.config.num_train_timesteps // self.num_inference_steps
    alpha_prod_t = self.alphas_cumprod[timestep.cpu()]
    alpha_prod_t_prev = torch.where(prev_timestep >= 0, self.alphas_cumprod
        [prev_timestep], self.final_alpha_cumprod).reshape(-1, 1, 1, 1)
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
        sqrt_alpha_prod = (alpha_prod_t ** 0.5).reshape(-1, 1, 1, 1).to(sample
            .device)
        sqer_beta_prod = (beta_prod_t ** 0.5).reshape(-1, 1, 1, 1).to(
            model_output.device)
        pred_original_sample = (sqrt_alpha_prod * sample - sqer_beta_prod *
            model_output)
        pred_epsilon = sqrt_alpha_prod * model_output + sqer_beta_prod * sample
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`'
            )
    if self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(-self.config.
            clip_sample_range, self.config.clip_sample_range)
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    variance = self._get_variance(timestep.cpu(), prev_timestep).reshape(-1,
        1, 1, 1)
    std_dev_t = eta * variance ** 0.5
    if use_clipped_model_output:
        pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample
            ) / beta_prod_t ** 0.5
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2).to(
        pred_epsilon.device) ** 0.5 * pred_epsilon
    prev_sample = alpha_prod_t_prev.to(pred_epsilon.device
        ) ** 0.5 * pred_original_sample + pred_sample_direction
    if eta > 0:
        if variance_noise is not None and generator is not None:
            raise ValueError(
                'Cannot pass both generator and variance_noise. Please make sure that either `generator` or `variance_noise` stays `None`.'
                )
        if variance_noise is None:
            variance_noise = randn_tensor(model_output.shape, generator=
                generator, device=model_output.device, dtype=model_output.dtype
                )
        variance = std_dev_t * variance_noise
        prev_sample = prev_sample + variance
    if not return_dict:
        return prev_sample,
    return DDIMSchedulerOutput(prev_sample=prev_sample,
        pred_original_sample=pred_original_sample)
