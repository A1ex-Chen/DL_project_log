def step_plms(self, model_output: torch.Tensor, timestep: int, sample:
    torch.Tensor, return_dict: bool=True) ->Union[SchedulerOutput, Tuple]:
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
    if not self.config.skip_prk_steps and len(self.ets) < 3:
        raise ValueError(
            f"{self.__class__} can only be run AFTER scheduler has been run in 'prk' mode for at least 12 iterations See: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_pndm.py for more information."
            )
    prev_timestep = (timestep - self.config.num_train_timesteps // self.
        num_inference_steps)
    if self.counter != 1:
        self.ets = self.ets[-3:]
        self.ets.append(model_output)
    else:
        prev_timestep = timestep
        timestep = (timestep + self.config.num_train_timesteps // self.
            num_inference_steps)
    if len(self.ets) == 1 and self.counter == 0:
        model_output = model_output
        self.cur_sample = sample
    elif len(self.ets) == 1 and self.counter == 1:
        model_output = (model_output + self.ets[-1]) / 2
        sample = self.cur_sample
        self.cur_sample = None
    elif len(self.ets) == 2:
        model_output = (3 * self.ets[-1] - self.ets[-2]) / 2
    elif len(self.ets) == 3:
        model_output = (23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.
            ets[-3]) / 12
    else:
        model_output = 1 / 24 * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 *
            self.ets[-3] - 9 * self.ets[-4])
    prev_sample = self._get_prev_sample(sample, timestep, prev_timestep,
        model_output)
    self.counter += 1
    if not return_dict:
        return prev_sample,
    return SchedulerOutput(prev_sample=prev_sample)
