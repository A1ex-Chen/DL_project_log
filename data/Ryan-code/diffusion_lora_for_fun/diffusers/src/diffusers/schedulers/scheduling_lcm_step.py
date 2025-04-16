def step(self, model_output: torch.Tensor, timestep: int, sample: torch.
    Tensor, generator: Optional[torch.Generator]=None, return_dict: bool=True
    ) ->Union[LCMSchedulerOutput, Tuple]:
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
                Whether or not to return a [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] or `tuple`.
        Returns:
            [`~schedulers.scheduling_utils.LCMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
    if self.step_index is None:
        self._init_step_index(timestep)
    prev_step_index = self.step_index + 1
    if prev_step_index < len(self.timesteps):
        prev_timestep = self.timesteps[prev_step_index]
    else:
        prev_timestep = timestep
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep
        ] if prev_timestep >= 0 else self.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)
    if self.config.prediction_type == 'epsilon':
        predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output
            ) / alpha_prod_t.sqrt()
    elif self.config.prediction_type == 'sample':
        predicted_original_sample = model_output
    elif self.config.prediction_type == 'v_prediction':
        predicted_original_sample = alpha_prod_t.sqrt(
            ) * sample - beta_prod_t.sqrt() * model_output
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or `v_prediction` for `LCMScheduler`.'
            )
    if self.config.thresholding:
        predicted_original_sample = self._threshold_sample(
            predicted_original_sample)
    elif self.config.clip_sample:
        predicted_original_sample = predicted_original_sample.clamp(-self.
            config.clip_sample_range, self.config.clip_sample_range)
    denoised = c_out * predicted_original_sample + c_skip * sample
    if self.step_index != self.num_inference_steps - 1:
        noise = randn_tensor(model_output.shape, generator=generator,
            device=model_output.device, dtype=denoised.dtype)
        prev_sample = alpha_prod_t_prev.sqrt(
            ) * denoised + beta_prod_t_prev.sqrt() * noise
    else:
        prev_sample = denoised
    self._step_index += 1
    if not return_dict:
        return prev_sample, denoised
    return LCMSchedulerOutput(prev_sample=prev_sample, denoised=denoised)
