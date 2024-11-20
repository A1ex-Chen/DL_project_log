def step_prk(self, model_output: torch.Tensor, timestep: int, sample: torch
    .Tensor, return_dict: bool=True) ->Union[SchedulerOutput, Tuple]:
    """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the Runge-Kutta method. It performs four forward passes to approximate the solution to the differential
        equation.

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
    diff_to_prev = (0 if self.counter % 2 else self.config.
        num_train_timesteps // self.num_inference_steps // 2)
    prev_timestep = timestep - diff_to_prev
    timestep = self.prk_timesteps[self.counter // 4 * 4]
    if self.counter % 4 == 0:
        self.cur_model_output += 1 / 6 * model_output
        self.ets.append(model_output)
        self.cur_sample = sample
    elif (self.counter - 1) % 4 == 0:
        self.cur_model_output += 1 / 3 * model_output
    elif (self.counter - 2) % 4 == 0:
        self.cur_model_output += 1 / 3 * model_output
    elif (self.counter - 3) % 4 == 0:
        model_output = self.cur_model_output + 1 / 6 * model_output
        self.cur_model_output = 0
    cur_sample = self.cur_sample if self.cur_sample is not None else sample
    prev_sample = self._get_prev_sample(cur_sample, timestep, prev_timestep,
        model_output)
    self.counter += 1
    if not return_dict:
        return prev_sample,
    return SchedulerOutput(prev_sample=prev_sample)
