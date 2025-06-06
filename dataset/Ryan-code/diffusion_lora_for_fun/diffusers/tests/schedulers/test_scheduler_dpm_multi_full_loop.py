def full_loop(self, scheduler=None, **config):
    if scheduler is None:
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)
    num_inference_steps = 10
    model = self.dummy_model()
    sample = self.dummy_sample_deter
    scheduler.set_timesteps(num_inference_steps)
    generator = torch.manual_seed(0)
    for i, t in enumerate(scheduler.timesteps):
        residual = model(sample, t)
        sample = scheduler.step(residual, t, sample, generator=generator
            ).prev_sample
    return sample
