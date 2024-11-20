def step(self, model_output: torch.Tensor, timeindex: int, timestep: int,
    sample: torch.Tensor, eta: float=0.0, use_clipped_model_output: bool=
    False, generator=None, variance_noise: Optional[torch.Tensor]=None,
    return_dict: bool=True) ->Union[LCMSchedulerOutput, Tuple]:
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
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
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
    prev_timeindex = timeindex + 1
    if prev_timeindex < len(self.timesteps):
        prev_timestep = self.timesteps[prev_timeindex]
    else:
        prev_timestep = timestep
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep
        ] if prev_timestep >= 0 else self.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)
    parameterization = self.config.prediction_type
    if parameterization == 'epsilon':
        pred_x0 = (sample - beta_prod_t.sqrt() * model_output
            ) / alpha_prod_t.sqrt()
    elif parameterization == 'sample':
        pred_x0 = model_output
    elif parameterization == 'v_prediction':
        pred_x0 = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt(
            ) * model_output
    denoised = c_out * pred_x0 + c_skip * sample
    if len(self.timesteps) > 1:
        noise = torch.randn(model_output.shape).to(model_output.device)
        prev_sample = alpha_prod_t_prev.sqrt(
            ) * denoised + beta_prod_t_prev.sqrt() * noise
    else:
        prev_sample = denoised
    if not return_dict:
        return prev_sample, denoised
    return LCMSchedulerOutput(prev_sample=prev_sample, denoised=denoised)
