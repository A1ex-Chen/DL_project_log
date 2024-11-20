def full_loop(self, num_inference_steps=10, seed=0, **config):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config(**config)
    scheduler = scheduler_class(**scheduler_config)
    eta = 0.0
    model = self.dummy_model()
    sample = self.dummy_sample_deter
    generator = torch.manual_seed(seed)
    scheduler.set_timesteps(num_inference_steps)
    for t in scheduler.timesteps:
        residual = model(sample, t)
        sample = scheduler.step(residual, t, sample, eta, generator
            ).prev_sample
    return sample
