def ddim_bit_scheduler_step(self, model_output: torch.Tensor, timestep: int,
    sample: torch.Tensor, eta: float=0.0, use_clipped_model_output: bool=
    True, generator=None, return_dict: bool=True) ->Union[
    DDIMSchedulerOutput, Tuple]:
    """
    Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
    process from the learned model outputs (most often the predicted noise).
    Args:
        model_output (`torch.Tensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.Tensor`):
            current instance of sample being created by diffusion process.
        eta (`float`): weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`): TODO
        generator: random number generator.
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
    prev_timestep = (timestep - self.config.num_train_timesteps // self.
        num_inference_steps)
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep
        ] if prev_timestep >= 0 else self.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output
        ) / alpha_prod_t ** 0.5
    scale = self.bit_scale
    if self.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -scale, scale)
    variance = self._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** 0.5
    if use_clipped_model_output:
        model_output = (sample - alpha_prod_t ** 0.5 * pred_original_sample
            ) / beta_prod_t ** 0.5
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2
        ) ** 0.5 * model_output
    prev_sample = (alpha_prod_t_prev ** 0.5 * pred_original_sample +
        pred_sample_direction)
    if eta > 0:
        device = model_output.device if torch.is_tensor(model_output
            ) else 'cpu'
        noise = torch.randn(model_output.shape, dtype=model_output.dtype,
            generator=generator).to(device)
        variance = self._get_variance(timestep, prev_timestep
            ) ** 0.5 * eta * noise
        prev_sample = prev_sample + variance
    if not return_dict:
        return prev_sample,
    return DDIMSchedulerOutput(prev_sample=prev_sample,
        pred_original_sample=pred_original_sample)
