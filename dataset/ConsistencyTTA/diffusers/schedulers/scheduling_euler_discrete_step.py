def step(self, model_output: torch.FloatTensor, timestep: Union[float,
    torch.FloatTensor], sample: torch.FloatTensor, s_churn: float=0.0,
    s_tmin: float=0.0, s_tmax: float=float('inf'), s_noise: float=1.0,
    generator: Optional[torch.Generator]=None, return_dict: bool=True) ->Union[
    EulerDiscreteSchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`float`): current timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            s_churn (`float`)
            s_tmin  (`float`)
            s_tmax  (`float`)
            s_noise (`float`)
            generator (`torch.Generator`, optional): Random number generator.
            return_dict (`bool`): option for returning tuple rather than EulerDiscreteSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.EulerDiscreteSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.EulerDiscreteSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.

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
    gamma = min(s_churn / (len(self.sigmas) - 1), 2 ** 0.5 - 1
        ) if s_tmin <= sigma <= s_tmax else 0.0
    noise = randn_tensor(model_output.shape, dtype=model_output.dtype,
        device=model_output.device, generator=generator)
    eps = noise * s_noise
    sigma_hat = sigma * (gamma + 1)
    if gamma > 0:
        sample = sample + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5
    if (self.config.prediction_type == 'original_sample' or self.config.
        prediction_type == 'sample'):
        pred_original_sample = model_output
    elif self.config.prediction_type == 'epsilon':
        pred_original_sample = sample - sigma_hat * model_output
    elif self.config.prediction_type == 'v_prediction':
        pred_original_sample = model_output * (-sigma / (sigma ** 2 + 1) ** 0.5
            ) + sample / (sigma ** 2 + 1)
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`'
            )
    derivative = (sample - pred_original_sample) / sigma_hat
    dt = self.sigmas[step_index + 1] - sigma_hat
    prev_sample = sample + derivative * dt
    if not return_dict:
        return prev_sample,
    return EulerDiscreteSchedulerOutput(prev_sample=prev_sample,
        pred_original_sample=pred_original_sample)
