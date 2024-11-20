def step(self, model_output: torch.FloatTensor, timestep: int, sample:
    torch.FloatTensor, return_dict: bool=True) ->Union[SchedulerOutput, Tuple]:
    """
        Step function propagating the sample with the singlestep DPM-Solver.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
    if isinstance(timestep, torch.Tensor):
        timestep = timestep.to(self.timesteps.device)
    step_index = (self.timesteps == timestep).nonzero()
    if len(step_index) == 0:
        step_index = len(self.timesteps) - 1
    else:
        step_index = step_index.item()
    prev_timestep = 0 if step_index == len(self.timesteps
        ) - 1 else self.timesteps[step_index + 1]
    model_output = self.convert_model_output(model_output, timestep, sample)
    for i in range(self.config.solver_order - 1):
        self.model_outputs[i] = self.model_outputs[i + 1]
    self.model_outputs[-1] = model_output
    order = self.order_list[step_index]
    while self.model_outputs[-order] is None:
        order -= 1
    if order == 1:
        self.sample = sample
    timestep_list = [self.timesteps[step_index - i] for i in range(order - 
        1, 0, -1)] + [timestep]
    prev_sample = self.singlestep_dpm_solver_update(self.model_outputs,
        timestep_list, prev_timestep, self.sample, order)
    if not return_dict:
        return prev_sample,
    return SchedulerOutput(prev_sample=prev_sample)
