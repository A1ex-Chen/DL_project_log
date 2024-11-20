def step(self, model_output: torch.FloatTensor, timestep: int, sample:
    torch.FloatTensor, return_dict: bool=True) ->Union[SchedulerOutput, Tuple]:
    """
        Step function propagating the sample with the multistep UniPC.

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
    use_corrector = (step_index > 0 and step_index - 1 not in self.
        disable_corrector and self.last_sample is not None)
    model_output_convert = self.convert_model_output(model_output, timestep,
        sample)
    if use_corrector:
        sample = self.multistep_uni_c_bh_update(this_model_output=
            model_output_convert, this_timestep=timestep, last_sample=self.
            last_sample, this_sample=sample, order=self.this_order)
    prev_timestep = 0 if step_index == len(self.timesteps
        ) - 1 else self.timesteps[step_index + 1]
    for i in range(self.config.solver_order - 1):
        self.model_outputs[i] = self.model_outputs[i + 1]
        self.timestep_list[i] = self.timestep_list[i + 1]
    self.model_outputs[-1] = model_output_convert
    self.timestep_list[-1] = timestep
    if self.config.lower_order_final:
        this_order = min(self.config.solver_order, len(self.timesteps) -
            step_index)
    else:
        this_order = self.config.solver_order
    self.this_order = min(this_order, self.lower_order_nums + 1)
    assert self.this_order > 0
    self.last_sample = sample
    prev_sample = self.multistep_uni_p_bh_update(model_output=model_output,
        prev_timestep=prev_timestep, sample=sample, order=self.this_order)
    if self.lower_order_nums < self.config.solver_order:
        self.lower_order_nums += 1
    if not return_dict:
        return prev_sample,
    return SchedulerOutput(prev_sample=prev_sample)
