def step(self, model_output: torch.Tensor, timestep: int, sample: torch.
    Tensor, generator=None, return_dict: bool=True) ->Union[
    DDPMWuerstchenSchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                current instance of sample being created by diffusion process.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than DDPMWuerstchenSchedulerOutput class

        Returns:
            [`DDPMWuerstchenSchedulerOutput`] or `tuple`: [`DDPMWuerstchenSchedulerOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
    dtype = model_output.dtype
    device = model_output.device
    t = timestep
    prev_t = self.previous_timestep(t)
    alpha_cumprod = self._alpha_cumprod(t, device).view(t.size(0), *[(1) for
        _ in sample.shape[1:]])
    alpha_cumprod_prev = self._alpha_cumprod(prev_t, device).view(prev_t.
        size(0), *[(1) for _ in sample.shape[1:]])
    alpha = alpha_cumprod / alpha_cumprod_prev
    mu = (1.0 / alpha).sqrt() * (sample - (1 - alpha) * model_output / (1 -
        alpha_cumprod).sqrt())
    std_noise = randn_tensor(mu.shape, generator=generator, device=
        model_output.device, dtype=model_output.dtype)
    std = ((1 - alpha) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        ).sqrt() * std_noise
    pred = mu + std * (prev_t != 0).float().view(prev_t.size(0), *[(1) for
        _ in sample.shape[1:]])
    if not return_dict:
        return pred.to(dtype),
    return DDPMWuerstchenSchedulerOutput(prev_sample=pred.to(dtype))
