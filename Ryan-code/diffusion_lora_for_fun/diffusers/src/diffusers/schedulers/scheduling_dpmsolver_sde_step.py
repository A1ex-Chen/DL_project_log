def step(self, model_output: Union[torch.Tensor, np.ndarray], timestep:
    Union[float, torch.Tensor], sample: Union[torch.Tensor, np.ndarray],
    return_dict: bool=True, s_noise: float=1.0) ->Union[SchedulerOutput, Tuple
    ]:
    """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor` or `np.ndarray`):
                The direct output from learned diffusion model.
            timestep (`float` or `torch.Tensor`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor` or `np.ndarray`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or tuple.
            s_noise (`float`, *optional*, defaults to 1.0):
                Scaling factor for noise added to the sample.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
    if self.step_index is None:
        self._init_step_index(timestep)
    if self.noise_sampler is None:
        min_sigma, max_sigma = self.sigmas[self.sigmas > 0].min(
            ), self.sigmas.max()
        self.noise_sampler = BrownianTreeNoiseSampler(sample, min_sigma,
            max_sigma, self.noise_sampler_seed)

    def sigma_fn(_t: torch.Tensor) ->torch.Tensor:
        return _t.neg().exp()

    def t_fn(_sigma: torch.Tensor) ->torch.Tensor:
        return _sigma.log().neg()
    if self.state_in_first_order:
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
    else:
        sigma = self.sigmas[self.step_index - 1]
        sigma_next = self.sigmas[self.step_index]
    midpoint_ratio = 0.5
    t, t_next = t_fn(sigma), t_fn(sigma_next)
    delta_time = t_next - t
    t_proposed = t + delta_time * midpoint_ratio
    if self.config.prediction_type == 'epsilon':
        sigma_input = sigma if self.state_in_first_order else sigma_fn(
            t_proposed)
        pred_original_sample = sample - sigma_input * model_output
    elif self.config.prediction_type == 'v_prediction':
        sigma_input = sigma if self.state_in_first_order else sigma_fn(
            t_proposed)
        pred_original_sample = model_output * (-sigma_input / (sigma_input **
            2 + 1) ** 0.5) + sample / (sigma_input ** 2 + 1)
    elif self.config.prediction_type == 'sample':
        raise NotImplementedError('prediction_type not implemented yet: sample'
            )
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`'
            )
    if sigma_next == 0:
        derivative = (sample - pred_original_sample) / sigma
        dt = sigma_next - sigma
        prev_sample = sample + derivative * dt
    else:
        if self.state_in_first_order:
            t_next = t_proposed
        else:
            sample = self.sample
        sigma_from = sigma_fn(t)
        sigma_to = sigma_fn(t_next)
        sigma_up = min(sigma_to, (sigma_to ** 2 * (sigma_from ** 2 - 
            sigma_to ** 2) / sigma_from ** 2) ** 0.5)
        sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
        ancestral_t = t_fn(sigma_down)
        prev_sample = sigma_fn(ancestral_t) / sigma_fn(t) * sample - (t -
            ancestral_t).expm1() * pred_original_sample
        prev_sample = prev_sample + self.noise_sampler(sigma_fn(t),
            sigma_fn(t_next)) * s_noise * sigma_up
        if self.state_in_first_order:
            self.sample = sample
            self.mid_point_sigma = sigma_fn(t_next)
        else:
            self.sample = None
            self.mid_point_sigma = None
    self._step_index += 1
    if not return_dict:
        return prev_sample,
    return SchedulerOutput(prev_sample=prev_sample)
