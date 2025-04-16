def add_noise(self, sample, timesteps, generator=None):
    step_idx = (self.timesteps == timesteps).nonzero()
    ratio = (step_idx + 1) / len(self.timesteps)
    if self.config.masking_schedule == 'cosine':
        mask_ratio = torch.cos(ratio * math.pi / 2)
    elif self.config.masking_schedule == 'linear':
        mask_ratio = 1 - ratio
    else:
        raise ValueError(
            f'unknown masking schedule {self.config.masking_schedule}')
    mask_indices = torch.rand(sample.shape, device=generator.device if 
        generator is not None else sample.device, generator=generator).to(
        sample.device) < mask_ratio
    masked_sample = sample.clone()
    masked_sample[mask_indices] = self.config.mask_token_id
    return masked_sample
