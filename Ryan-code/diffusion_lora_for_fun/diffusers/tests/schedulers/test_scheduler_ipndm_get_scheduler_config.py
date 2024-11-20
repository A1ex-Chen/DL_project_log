def get_scheduler_config(self, **kwargs):
    config = {'num_train_timesteps': 1000}
    config.update(**kwargs)
    return config
