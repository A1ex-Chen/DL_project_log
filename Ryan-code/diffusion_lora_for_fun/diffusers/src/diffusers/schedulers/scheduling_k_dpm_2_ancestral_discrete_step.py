def step(self, model_output: Union[torch.Tensor, np.ndarray], timestep:
    Union[float, torch.Tensor], sample: Union[torch.Tensor, np.ndarray],
    generator: Optional[torch.Generator]=None, return_dict: bool=True) ->Union[
    SchedulerOutput, Tuple]:
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
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
    if self.step_index is None:
        self._init_step_index(timestep)
    if self.state_in_first_order:
        sigma = self.sigmas[self.step_index]
        sigma_interpol = self.sigmas_interpol[self.step_index]
        sigma_up = self.sigmas_up[self.step_index]
        sigma_down = self.sigmas_down[self.step_index - 1]
    else:
        sigma = self.sigmas[self.step_index - 1]
        sigma_interpol = self.sigmas_interpol[self.step_index - 1]
        sigma_up = self.sigmas_up[self.step_index - 1]
        sigma_down = self.sigmas_down[self.step_index - 1]
    gamma = 0
    sigma_hat = sigma * (gamma + 1)
    device = model_output.device
    noise = randn_tensor(model_output.shape, dtype=model_output.dtype,
        device=device, generator=generator)
    if self.config.prediction_type == 'epsilon':
        sigma_input = (sigma_hat if self.state_in_first_order else
            sigma_interpol)
        pred_original_sample = sample - sigma_input * model_output
    elif self.config.prediction_type == 'v_prediction':
        sigma_input = (sigma_hat if self.state_in_first_order else
            sigma_interpol)
        pred_original_sample = model_output * (-sigma_input / (sigma_input **
            2 + 1) ** 0.5) + sample / (sigma_input ** 2 + 1)
    elif self.config.prediction_type == 'sample':
        raise NotImplementedError('prediction_type not implemented yet: sample'
            )
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`'
            )
    if self.state_in_first_order:
        derivative = (sample - pred_original_sample) / sigma_hat
        dt = sigma_interpol - sigma_hat
        self.sample = sample
        self.dt = dt
        prev_sample = sample + derivative * dt
    else:
        derivative = (sample - pred_original_sample) / sigma_interpol
        dt = sigma_down - sigma_hat
        sample = self.sample
        self.sample = None
        prev_sample = sample + derivative * dt
        prev_sample = prev_sample + noise * sigma_up
    self._step_index += 1
    if not return_dict:
        return prev_sample,
    return SchedulerOutput(prev_sample=prev_sample)
