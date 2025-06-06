def full_loop(self, **config):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config(**config)
    scheduler = scheduler_class(**scheduler_config)
    num_inference_steps = self.num_inference_steps
    scheduler.set_timesteps(num_inference_steps)
    model = self.dummy_model()
    sample = self.dummy_sample_deter * scheduler.init_noise_sigma
    sample = sample.to(torch_device)
    for i, t in enumerate(scheduler.timesteps):
        sample = scheduler.scale_model_input(sample, t)
        model_output = model(sample, t)
        output = scheduler.step(model_output, t, sample)
        sample = output.prev_sample
    return sample
