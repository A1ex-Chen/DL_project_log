def test_custom_timesteps_passing_both_num_inference_steps_and_timesteps(self):
    scheduler_class = self.scheduler_classes[0]
    scheduler_config = self.get_scheduler_config()
    scheduler = scheduler_class(**scheduler_config)
    timesteps = [100, 87, 50, 1, 0]
    num_inference_steps = len(timesteps)
    with self.assertRaises(ValueError, msg=
        'Can only pass one of `num_inference_steps` or `custom_timesteps`.'):
        scheduler.set_timesteps(num_inference_steps=num_inference_steps,
            timesteps=timesteps)
