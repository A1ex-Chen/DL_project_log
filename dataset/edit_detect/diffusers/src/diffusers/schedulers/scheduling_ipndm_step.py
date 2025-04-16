def step(self, model_output: torch.Tensor, timestep: int, sample: torch.
    Tensor, return_dict: bool=True) ->Union[SchedulerOutput, Tuple]:
    """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the linear multistep method. It performs one forward pass multiple times to approximate the solution.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
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
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
    if self.step_index is None:
        self._init_step_index(timestep)
    timestep_index = self.step_index
    prev_timestep_index = self.step_index + 1
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
    self._step_index += 1
    if not return_dict:
        return prev_sample,
    return SchedulerOutput(prev_sample=prev_sample)
