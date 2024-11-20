def step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep:
    Union[float, torch.FloatTensor], sample: Union[torch.FloatTensor, np.
    ndarray], return_dict: bool=True) ->Union[SchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. 
        Core function to propagate the diffusion process from the learned 
        model outputs (most often the predicted noise).
        Args:
            model_output (`torch.FloatTensor` or `np.ndarray`): 
                direct output from learned diffusion model. 
            timestep (`int`): 
                current discrete timestep in the diffusion chain. 
            sample (`torch.FloatTensor` or `np.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): 
                option for returning tuple rather than SchedulerOutput class
        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.SchedulerOutput`] if `return_dict` 
                is True, otherwise a `tuple`. When returning a tuple,
                the first element is the sample tensor.
        """
    if not torch.is_tensor(timestep):
        timestep = torch.tensor(timestep)
    timestep = timestep.reshape(-1).to(sample.device)
    step_index = self.index_for_timestep(timestep)
    if self.state_in_first_order:
        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]
    else:
        sigma = self.sigmas[step_index - 1]
        sigma_next = self.sigmas[step_index]
    sigma = sigma.reshape(-1, 1, 1, 1).to(sample.device)
    sigma_next = sigma_next.reshape(-1, 1, 1, 1).to(sample.device)
    sigma_input = sigma if self.state_in_first_order else sigma_next
    if self.config.prediction_type == 'epsilon':
        pred_original_sample = sample - sigma_input * model_output
    elif self.config.prediction_type == 'v_prediction':
        alpha_prod = 1 / (sigma_input ** 2 + 1)
        pred_original_sample = sample * alpha_prod - model_output * (
            sigma_input * alpha_prod ** 0.5)
    elif self.config.prediction_type == 'sample':
        raise NotImplementedError('prediction_type not implemented yet: sample'
            )
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`'
            )
    if self.state_in_first_order:
        derivative = (sample - pred_original_sample) / sigma
        dt = sigma_next - sigma
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
    if not return_dict:
        return prev_sample,
    return SchedulerOutput(prev_sample=prev_sample)
