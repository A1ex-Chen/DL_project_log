def step(self, model_output: torch.Tensor, timestep: int, sample: torch.
    Tensor, generator=None, return_dict: bool=True) ->Union[SchedulerOutput,
    Tuple]:
    """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the multistep DPMSolver.

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
    lower_order_final = self.step_index == len(self.timesteps) - 1 and (self
        .config.euler_at_final or self.config.lower_order_final and len(
        self.timesteps) < 15 or self.config.final_sigmas_type == 'zero')
    lower_order_second = self.step_index == len(self.timesteps
        ) - 2 and self.config.lower_order_final and len(self.timesteps) < 15
    model_output = self.convert_model_output(model_output, sample=sample)
    for i in range(self.config.solver_order - 1):
        self.model_outputs[i] = self.model_outputs[i + 1]
    self.model_outputs[-1] = model_output
    if self.config.algorithm_type == 'sde-dpmsolver++':
        noise = randn_tensor(model_output.shape, generator=generator,
            device=model_output.device, dtype=model_output.dtype)
    else:
        noise = None
    if (self.config.solver_order == 1 or self.lower_order_nums < 1 or
        lower_order_final):
        prev_sample = self.dpm_solver_first_order_update(model_output,
            sample=sample, noise=noise)
    elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
        prev_sample = self.multistep_dpm_solver_second_order_update(self.
            model_outputs, sample=sample, noise=noise)
    else:
        prev_sample = self.multistep_dpm_solver_third_order_update(self.
            model_outputs, sample=sample)
    if self.lower_order_nums < self.config.solver_order:
        self.lower_order_nums += 1
    self._step_index += 1
    if not return_dict:
        return prev_sample,
    return SchedulerOutput(prev_sample=prev_sample)
