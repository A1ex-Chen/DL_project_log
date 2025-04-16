def full_loop(self, **config):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config(**config)
    scheduler = scheduler_class(**scheduler_config)
    state = scheduler.create_state()
    num_inference_steps = 10
    model = self.dummy_model()
    sample = self.dummy_sample_deter
    state = scheduler.set_timesteps(state, num_inference_steps, shape=
        sample.shape)
    for i, t in enumerate(state.prk_timesteps):
        residual = model(sample, t)
        sample, state = scheduler.step_prk(state, residual, t, sample)
    for i, t in enumerate(state.plms_timesteps):
        residual = model(sample, t)
        sample, state = scheduler.step_plms(state, residual, t, sample)
    return sample
