def step(self, model_output: torch.Tensor, timestep: Union[float, torch.
    Tensor], sample: torch.Tensor, order: int=4, return_dict: bool=True
    ) ->Union[LMSDiscreteSchedulerOutput, Tuple]:
    """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float` or `torch.Tensor`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            order (`int`, defaults to 4):
                The order of the linear multistep method.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
    if not self.is_scale_input_called:
        warnings.warn(
            'The `scale_model_input` function should be called before `step` to ensure correct denoising. See `StableDiffusionPipeline` for a usage example.'
            )
    if self.step_index is None:
        self._init_step_index(timestep)
    sigma = self.sigmas[self.step_index]
    if self.config.prediction_type == 'epsilon':
        pred_original_sample = sample - sigma * model_output
    elif self.config.prediction_type == 'v_prediction':
        pred_original_sample = model_output * (-sigma / (sigma ** 2 + 1) ** 0.5
            ) + sample / (sigma ** 2 + 1)
    elif self.config.prediction_type == 'sample':
        pred_original_sample = model_output
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`'
            )
    derivative = (sample - pred_original_sample) / sigma
    self.derivatives.append(derivative)
    if len(self.derivatives) > order:
        self.derivatives.pop(0)
    order = min(self.step_index + 1, order)
    lms_coeffs = [self.get_lms_coefficient(order, self.step_index,
        curr_order) for curr_order in range(order)]
    prev_sample = sample + sum(coeff * derivative for coeff, derivative in
        zip(lms_coeffs, reversed(self.derivatives)))
    self._step_index += 1
    if not return_dict:
        return prev_sample,
    return LMSDiscreteSchedulerOutput(prev_sample=prev_sample,
        pred_original_sample=pred_original_sample)
