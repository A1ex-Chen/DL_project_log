def step(self, model_output: torch.FloatTensor, timestep: int, sample:
    torch.FloatTensor, return_dict: bool=True) ->Union[SchedulerOutput, Tuple]:
    """
        Step function propagating the sample with the linear multi-step method. This has one forward pass with multiple
        times to approximate the solution.

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
    timestep_index = (self.timesteps == timestep).nonzero().item()
    prev_timestep_index = timestep_index + 1
    ets = sample * self.betas[timestep_index] + model_output * self.alphas[
        timestep_index]
    self.ets.append(ets)
    if len(self.ets) == 1:
        ets = self.ets[-1]
    elif len(self.ets) == 2:
        ets = (3 * self.ets[-1] - self.ets[-2]) / 2
    elif len(self.ets) == 3:
        ets = (23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]) / 12
    else:
        ets = 1 / 24 * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.
            ets[-3] - 9 * self.ets[-4])
    prev_sample = self._get_prev_sample(sample, timestep_index,
        prev_timestep_index, ets)
    if not return_dict:
        return prev_sample,
    return SchedulerOutput(prev_sample=prev_sample)
