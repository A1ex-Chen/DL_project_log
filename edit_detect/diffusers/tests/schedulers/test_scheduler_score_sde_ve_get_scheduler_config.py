def get_scheduler_config(self, **kwargs):
    config = {'num_train_timesteps': 2000, 'snr': 0.15, 'sigma_min': 0.01,
        'sigma_max': 1348, 'sampling_eps': 1e-05}
    config.update(**kwargs)
    return config
