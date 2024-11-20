def get_scheduler_config(self, **kwargs):
    config = {'num_train_timesteps': 1100, 'beta_start': 0.0001, 'beta_end':
        0.02, 'beta_schedule': 'linear', 'noise_sampler_seed': 0}
    config.update(**kwargs)
    return config
