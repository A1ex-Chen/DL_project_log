def step(self, model_output: torch.Tensor, timestep: int, sample: torch.
    Tensor, return_dict: bool=True) ->Union[SchedulerOutput, Tuple]:
    """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise), and calls [`~PNDMScheduler.step_prk`]
        or [`~PNDMScheduler.step_plms`] depending on the internal variable `counter`.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
    if self.counter < len(self.prk_timesteps
        ) and not self.config.skip_prk_steps:
        return self.step_prk(model_output=model_output, timestep=timestep,
            sample=sample, return_dict=return_dict)
    else:
        return self.step_plms(model_output=model_output, timestep=timestep,
            sample=sample, return_dict=return_dict)
