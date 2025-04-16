def step(self, model_output: torch.FloatTensor, timestep: int, sample:
    torch.FloatTensor, return_dict: bool=True) ->Union[SchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        This function calls `step_prk()` or `step_plms()` depending on the internal variable `counter`.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.SchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
    if self.counter < len(self.prk_timesteps
        ) and not self.config.skip_prk_steps:
        return self.step_prk(model_output=model_output, timestep=timestep,
            sample=sample, return_dict=return_dict)
    else:
        return self.step_plms(model_output=model_output, timestep=timestep,
            sample=sample, return_dict=return_dict)
