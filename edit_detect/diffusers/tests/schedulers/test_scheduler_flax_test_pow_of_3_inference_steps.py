def test_pow_of_3_inference_steps(self):
    num_inference_steps = 27
    for scheduler_class in self.scheduler_classes:
        sample, _ = self.dummy_sample
        residual = 0.1 * sample
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        state = scheduler.create_state()
        state = scheduler.set_timesteps(state, num_inference_steps, shape=
            sample.shape)
        for i, t in enumerate(state.prk_timesteps[:2]):
            sample, state = scheduler.step_prk(state, residual, t, sample)
