def test_duplicated_timesteps(self, **config):
    for scheduler_class in self.scheduler_classes:
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)
        assert len(scheduler.timesteps) == scheduler.num_inference_steps
