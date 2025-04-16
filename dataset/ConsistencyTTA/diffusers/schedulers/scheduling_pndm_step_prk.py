def step_prk(self, model_output: torch.FloatTensor, timestep: int, sample:
    torch.FloatTensor, return_dict: bool=True) ->Union[SchedulerOutput, Tuple]:
    """
        Step function propagating the sample with the Runge-Kutta method. RK takes 4 forward passes to approximate the
        solution to the differential equation.

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
