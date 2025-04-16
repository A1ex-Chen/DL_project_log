def step_pred(self, model_output: torch.FloatTensor, timestep: int, sample:
    torch.FloatTensor, generator: Optional[torch.Generator]=None,
    return_dict: bool=True) ->Union[SdeVeOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~schedulers.scheduling_sde_ve.SdeVeOutput`] or `tuple`: [`~schedulers.scheduling_sde_ve.SdeVeOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
    if self.timesteps is None:
        raise ValueError(
            "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )
    timestep = timestep * torch.ones(sample.shape[0], device=sample.device)
    timesteps = (timestep * (len(self.timesteps) - 1)).long()
    timesteps = timesteps.to(self.discrete_sigmas.device)
    sigma = self.discrete_sigmas[timesteps].to(sample.device)
    adjacent_sigma = self.get_adjacent_sigma(timesteps, timestep).to(sample
        .device)
    drift = torch.zeros_like(sample)
    diffusion = (sigma ** 2 - adjacent_sigma ** 2) ** 0.5
    diffusion = diffusion.flatten()
    while len(diffusion.shape) < len(sample.shape):
        diffusion = diffusion.unsqueeze(-1)
    drift = drift - diffusion ** 2 * model_output
    noise = randn_tensor(sample.shape, layout=sample.layout, generator=
        generator, device=sample.device, dtype=sample.dtype)
    prev_sample_mean = sample - drift
    prev_sample = prev_sample_mean + diffusion * noise
    if not return_dict:
        return prev_sample, prev_sample_mean
    return SdeVeOutput(prev_sample=prev_sample, prev_sample_mean=
        prev_sample_mean)
