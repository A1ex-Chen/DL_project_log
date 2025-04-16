def full_loop_custom_timesteps(self, **config):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config(**config)
    scheduler = scheduler_class(**scheduler_config)
    num_inference_steps = 10
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps
    scheduler = scheduler_class(**scheduler_config)
    scheduler.set_timesteps(num_inference_steps=None, timesteps=timesteps)
    model = self.dummy_model()
    sample = self.dummy_sample_deter
    for i, t in enumerate(scheduler.timesteps):
        residual = model(sample, t)
        sample = scheduler.step(residual, t, sample).prev_sample
    return sample
