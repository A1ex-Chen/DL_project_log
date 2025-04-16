def step(self, model_output: Union[torch.Tensor, np.ndarray], timestep:
    Union[float, torch.Tensor], sample: Union[torch.Tensor, np.ndarray],
    return_dict: bool=True) ->Union[SchedulerOutput, Tuple]:
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
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
    if self.step_index is None:
        self._init_step_index(timestep)
    if self.state_in_first_order:
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
    else:
        sigma = self.sigmas[self.step_index - 1]
        sigma_next = self.sigmas[self.step_index]
    gamma = 0
    sigma_hat = sigma * (gamma + 1)
    if self.config.prediction_type == 'epsilon':
        sigma_input = sigma_hat if self.state_in_first_order else sigma_next
        pred_original_sample = sample - sigma_input * model_output
    elif self.config.prediction_type == 'v_prediction':
        sigma_input = sigma_hat if self.state_in_first_order else sigma_next
        pred_original_sample = model_output * (-sigma_input / (sigma_input **
            2 + 1) ** 0.5) + sample / (sigma_input ** 2 + 1)
    elif self.config.prediction_type == 'sample':
        pred_original_sample = model_output
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`'
            )
    if self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(-self.config.
            clip_sample_range, self.config.clip_sample_range)
    if self.state_in_first_order:
        derivative = (sample - pred_original_sample) / sigma_hat
        dt = sigma_next - sigma_hat
        self.prev_derivative = derivative
        self.dt = dt
        self.sample = sample
    else:
        derivative = (sample - pred_original_sample) / sigma_next
        derivative = (self.prev_derivative + derivative) / 2
        dt = self.dt
        sample = self.sample
        self.prev_derivative = None
        self.dt = None
        self.sample = None
    prev_sample = sample + derivative * dt
    self._step_index += 1
    if not return_dict:
        return prev_sample,
    return SchedulerOutput(prev_sample=prev_sample)
