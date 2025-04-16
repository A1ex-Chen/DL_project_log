def step(self, model_output: torch.Tensor, timestep: Union[float, torch.
    Tensor], sample: torch.Tensor, generator: Optional[torch.Generator]=
    None, return_dict: bool=True) ->Union[
    EulerAncestralDiscreteSchedulerOutput, Tuple]:
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
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """
    if isinstance(timestep, int) or isinstance(timestep, torch.IntTensor
        ) or isinstance(timestep, torch.LongTensor):
        raise ValueError(
            'Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to `EulerDiscreteScheduler.step()` is not supported. Make sure to pass one of the `scheduler.timesteps` as a timestep.'
            )
    if not self.is_scale_input_called:
        logger.warning(
            'The `scale_model_input` function should be called before `step` to ensure correct denoising. See `StableDiffusionPipeline` for a usage example.'
            )
    if self.step_index is None:
        self._init_step_index(timestep)
    sigma = self.sigmas[self.step_index]
    sample = sample.to(torch.float32)
    if self.config.prediction_type == 'epsilon':
        pred_original_sample = sample - sigma * model_output
    elif self.config.prediction_type == 'v_prediction':
        pred_original_sample = model_output * (-sigma / (sigma ** 2 + 1) ** 0.5
            ) + sample / (sigma ** 2 + 1)
    elif self.config.prediction_type == 'sample':
        raise NotImplementedError('prediction_type not implemented yet: sample'
            )
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`'
            )
    sigma_from = self.sigmas[self.step_index]
    sigma_to = self.sigmas[self.step_index + 1]
    sigma_up = (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / 
        sigma_from ** 2) ** 0.5
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    derivative = (sample - pred_original_sample) / sigma
    dt = sigma_down - sigma
    prev_sample = sample + derivative * dt
    device = model_output.device
    noise = randn_tensor(model_output.shape, dtype=model_output.dtype,
        device=device, generator=generator)
    prev_sample = prev_sample + noise * sigma_up
    prev_sample = prev_sample.to(model_output.dtype)
    self._step_index += 1
    if not return_dict:
        return prev_sample,
    return EulerAncestralDiscreteSchedulerOutput(prev_sample=prev_sample,
        pred_original_sample=pred_original_sample)
