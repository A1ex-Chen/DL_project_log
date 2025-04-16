def step(self, model_output: torch.Tensor, timestep: Union[float, torch.
    Tensor], sample: torch.Tensor, generator: Optional[torch.Generator]=
    None, return_dict: bool=True) ->Union[
    CMStochasticIterativeSchedulerOutput, Tuple]:
    """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            timestep (`float`):
                The current timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a
                [`~schedulers.scheduling_consistency_models.CMStochasticIterativeSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_consistency_models.CMStochasticIterativeSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_consistency_models.CMStochasticIterativeSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """
    if isinstance(timestep, int) or isinstance(timestep, torch.IntTensor
        ) or isinstance(timestep, torch.LongTensor):
        raise ValueError(
            f'Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to `{self.__class__}.step()` is not supported. Make sure to pass one of the `scheduler.timesteps` as a timestep.'
            )
    if not self.is_scale_input_called:
        logger.warning(
            'The `scale_model_input` function should be called before `step` to ensure correct denoising. See `StableDiffusionPipeline` for a usage example.'
            )
    sigma_min = self.config.sigma_min
    sigma_max = self.config.sigma_max
    if self.step_index is None:
        self._init_step_index(timestep)
    sigma = self.sigmas[self.step_index]
    if self.step_index + 1 < self.config.num_train_timesteps:
        sigma_next = self.sigmas[self.step_index + 1]
    else:
        sigma_next = self.sigmas[-1]
    c_skip, c_out = self.get_scalings_for_boundary_condition(sigma)
    denoised = c_out * model_output + c_skip * sample
    if self.config.clip_denoised:
        denoised = denoised.clamp(-1, 1)
    if len(self.timesteps) > 1:
        noise = randn_tensor(model_output.shape, dtype=model_output.dtype,
            device=model_output.device, generator=generator)
    else:
        noise = torch.zeros_like(model_output)
    z = noise * self.config.s_noise
    sigma_hat = sigma_next.clamp(min=sigma_min, max=sigma_max)
    prev_sample = denoised + z * (sigma_hat ** 2 - sigma_min ** 2) ** 0.5
    self._step_index += 1
    if not return_dict:
        return prev_sample,
    return CMStochasticIterativeSchedulerOutput(prev_sample=prev_sample)
