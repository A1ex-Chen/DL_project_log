def step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep:
    Union[float, torch.FloatTensor], sample: Union[torch.FloatTensor, np.
    ndarray], generator: Optional[torch.Generator]=None, return_dict: bool=True
    ) ->Union[SchedulerOutput, Tuple]:
    """
        Args:
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).
            model_output (`torch.FloatTensor` or `np.ndarray`): direct output from learned diffusion model. timestep
            (`int`): current discrete timestep in the diffusion chain. sample (`torch.FloatTensor` or `np.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class
        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.SchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
    step_index = self.index_for_timestep(timestep)
    if self.state_in_first_order:
        sigma = self.sigmas[step_index]
        sigma_interpol = self.sigmas_interpol[step_index]
        sigma_up = self.sigmas_up[step_index]
        sigma_down = self.sigmas_down[step_index - 1]
    else:
        sigma = self.sigmas[step_index - 1]
        sigma_interpol = self.sigmas_interpol[step_index - 1]
        sigma_up = self.sigmas_up[step_index - 1]
        sigma_down = self.sigmas_down[step_index - 1]
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
    if not return_dict:
        return prev_sample,
    return SchedulerOutput(prev_sample=prev_sample)
