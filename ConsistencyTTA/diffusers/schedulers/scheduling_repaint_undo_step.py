def undo_step(self, sample, timestep, generator=None):
    n = self.config.num_train_timesteps // self.num_inference_steps
    for i in range(n):
        beta = self.betas[timestep + i]
        if sample.device.type == 'mps':
            noise = randn_tensor(sample.shape, dtype=sample.dtype,
                generator=generator)
            noise = noise.to(sample.device)
        else:
            noise = randn_tensor(sample.shape, generator=generator, device=
                sample.device, dtype=sample.dtype)
        sample = (1 - beta) ** 0.5 * sample + beta ** 0.5 * noise
    return sample
