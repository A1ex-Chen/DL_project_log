def test_time_indices(self):
    kwargs = dict(self.forward_default_kwargs)
    num_inference_steps = kwargs.pop('num_inference_steps', None)
    scheduler_config = self.get_scheduler_config()
    scheduler = self.scheduler_classes[0](**scheduler_config)
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps
    for t in timesteps:
        self.check_over_forward(time_step=t)
