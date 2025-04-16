def step(self, model_output: torch.FloatTensor, timestep: Union[float,
    torch.FloatTensor], sample: torch.FloatTensor, order: int=4,
    return_dict: bool=True) ->Union[LMSDiscreteSchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`float`): current timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            order: coefficient for multi-step inference.
            return_dict (`bool`): option for returning tuple rather than LMSDiscreteSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.LMSDiscreteSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.LMSDiscreteSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.

        """
    if not self.is_scale_input_called:
        warnings.warn(
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
        pred_original_sample = model_output
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`'
            )
    derivative = (sample - pred_original_sample) / sigma
    self.derivatives.append(derivative)
    if len(self.derivatives) > order:
        self.derivatives.pop(0)
    order = min(step_index + 1, order)
    lms_coeffs = [self.get_lms_coefficient(order, step_index, curr_order) for
        curr_order in range(order)]
    prev_sample = sample + sum(coeff * derivative for coeff, derivative in
        zip(lms_coeffs, reversed(self.derivatives)))
    if not return_dict:
        return prev_sample,
    return LMSDiscreteSchedulerOutput(prev_sample=prev_sample,
        pred_original_sample=pred_original_sample)
