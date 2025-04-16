def step(self, model_output: torch.FloatTensor, timestep: Union[float,
    torch.FloatTensor], sample: torch.FloatTensor, generator: Optional[
    torch.Generator]=None, return_dict: bool=True) ->Union[
    EulerAncestralDiscreteSchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`float`): current timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            generator (`torch.Generator`, optional): Random number generator.
            return_dict (`bool`): option for returning tuple rather than EulerAncestralDiscreteSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.EulerAncestralDiscreteSchedulerOutput`] if `return_dict` is True, otherwise
            a `tuple`. When returning a tuple, the first element is the sample tensor.

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
    if isinstance(timestep, torch.Tensor):
        timestep = timestep.to(self.timesteps.device)
    step_index = (self.timesteps == timestep).nonzero().item()
    sigma = self.sigmas[step_index]
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
    sigma_from = self.sigmas[step_index]
    sigma_to = self.sigmas[step_index + 1]
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
    if not return_dict:
        return prev_sample,
    return EulerAncestralDiscreteSchedulerOutput(prev_sample=prev_sample,
        pred_original_sample=pred_original_sample)
