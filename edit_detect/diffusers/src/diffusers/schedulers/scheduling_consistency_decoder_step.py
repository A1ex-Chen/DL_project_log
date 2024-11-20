def step(self, model_output: torch.Tensor, timestep: Union[float, torch.
    Tensor], sample: torch.Tensor, generator: Optional[torch.Generator]=
    None, return_dict: bool=True) ->Union[ConsistencyDecoderSchedulerOutput,
    Tuple]:
    """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            timestep (`float`):
                The current timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a
                [`~schedulers.scheduling_consistency_models.ConsistencyDecoderSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_consistency_models.ConsistencyDecoderSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_consistency_models.ConsistencyDecoderSchedulerOutput`] is returned, otherwise
                a tuple is returned where the first element is the sample tensor.
        """
    x_0 = self.c_out[timestep] * model_output + self.c_skip[timestep] * sample
    timestep_idx = torch.where(self.timesteps == timestep)[0]
    if timestep_idx == len(self.timesteps) - 1:
        prev_sample = x_0
    else:
        noise = randn_tensor(x_0.shape, generator=generator, dtype=x_0.
            dtype, device=x_0.device)
        prev_sample = self.sqrt_alphas_cumprod[self.timesteps[timestep_idx + 1]
            ].to(x_0.dtype) * x_0 + self.sqrt_one_minus_alphas_cumprod[self
            .timesteps[timestep_idx + 1]].to(x_0.dtype) * noise
    if not return_dict:
        return prev_sample,
    return ConsistencyDecoderSchedulerOutput(prev_sample=prev_sample)
