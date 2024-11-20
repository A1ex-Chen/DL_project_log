def get_scheduler_config(self, **kwargs):
    config = {'num_train_timesteps': 1000, 'variance_type':
        'fixed_small_log', 'clip_sample': True, 'clip_sample_range': 1.0,
        'prediction_type': 'epsilon'}
    config.update(**kwargs)
    return config
