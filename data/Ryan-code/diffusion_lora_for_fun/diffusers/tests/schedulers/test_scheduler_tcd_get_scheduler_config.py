def get_scheduler_config(self, **kwargs):
    config = {'num_train_timesteps': 1000, 'beta_start': 0.00085,
        'beta_end': 0.012, 'beta_schedule': 'scaled_linear',
        'prediction_type': 'epsilon'}
    config.update(**kwargs)
    return config
