def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps:
    torch.Tensor) ->torch.Tensor:
    if isinstance(timesteps, int) or isinstance(timesteps, torch.IntTensor
        ) or isinstance(timesteps, torch.LongTensor):
        raise ValueError(
            'Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to `EulerDiscreteScheduler.get_velocity()` is not supported. Make sure to pass one of the `scheduler.timesteps` as a timestep.'
            )
    if sample.device.type == 'mps' and torch.is_floating_point(timesteps):
        schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.
            float32)
        timesteps = timesteps.to(sample.device, dtype=torch.float32)
    else:
        schedule_timesteps = self.timesteps.to(sample.device)
        timesteps = timesteps.to(sample.device)
    step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in
        timesteps]
    alphas_cumprod = self.alphas_cumprod.to(sample)
    sqrt_alpha_prod = alphas_cumprod[step_indices] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(sample.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[step_indices]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
    return velocity
