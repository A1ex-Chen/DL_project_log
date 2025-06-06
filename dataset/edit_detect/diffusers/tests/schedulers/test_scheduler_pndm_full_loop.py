def full_loop(self, **config):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config(**config)
    scheduler = scheduler_class(**scheduler_config)
    num_inference_steps = 10
    model = self.dummy_model()
    sample = self.dummy_sample_deter
    scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(scheduler.prk_timesteps):
        residual = model(sample, t)
        sample = scheduler.step_prk(residual, t, sample).prev_sample
    for i, t in enumerate(scheduler.plms_timesteps):
        residual = model(sample, t)
        sample = scheduler.step_plms(residual, t, sample).prev_sample
    return sample
