def get_scheduler_config(self, **kwargs):
    config = {'num_train_timesteps': 1000, 'beta_start': 0.0001, 'beta_end':
        0.02, 'beta_schedule': 'linear'}
    config.update(**kwargs)
    return config
