def step(self, model_output: torch.Tensor, timestep: int, sample: torch.
    Tensor, generator=None, return_dict: bool=True) ->Union[SchedulerOutput,
    Tuple]:
    """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the SA-Solver.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
    if self.step_index is None:
        self._init_step_index(timestep)
    use_corrector = self.step_index > 0 and self.last_sample is not None
    model_output_convert = self.convert_model_output(model_output, sample=
        sample)
    if use_corrector:
        current_tau = self.tau_func(self.timestep_list[-1])
        sample = self.stochastic_adams_moulton_update(this_model_output=
            model_output_convert, last_sample=self.last_sample, last_noise=
            self.last_noise, this_sample=sample, order=self.
            this_corrector_order, tau=current_tau)
    for i in range(max(self.config.predictor_order, self.config.
        corrector_order - 1) - 1):
        self.model_outputs[i] = self.model_outputs[i + 1]
        self.timestep_list[i] = self.timestep_list[i + 1]
    self.model_outputs[-1] = model_output_convert
    self.timestep_list[-1] = timestep
    noise = randn_tensor(model_output.shape, generator=generator, device=
        model_output.device, dtype=model_output.dtype)
    if self.config.lower_order_final:
        this_predictor_order = min(self.config.predictor_order, len(self.
            timesteps) - self.step_index)
        this_corrector_order = min(self.config.corrector_order, len(self.
            timesteps) - self.step_index + 1)
    else:
        this_predictor_order = self.config.predictor_order
        this_corrector_order = self.config.corrector_order
    self.this_predictor_order = min(this_predictor_order, self.
        lower_order_nums + 1)
    self.this_corrector_order = min(this_corrector_order, self.
        lower_order_nums + 2)
    assert self.this_predictor_order > 0
    assert self.this_corrector_order > 0
    self.last_sample = sample
    self.last_noise = noise
    current_tau = self.tau_func(self.timestep_list[-1])
    prev_sample = self.stochastic_adams_bashforth_update(model_output=
        model_output_convert, sample=sample, noise=noise, order=self.
        this_predictor_order, tau=current_tau)
    if self.lower_order_nums < max(self.config.predictor_order, self.config
        .corrector_order - 1):
        self.lower_order_nums += 1
    self._step_index += 1
    if not return_dict:
        return prev_sample,
    return SchedulerOutput(prev_sample=prev_sample)
