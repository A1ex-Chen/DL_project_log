def get_scheduler_config(self, **kwargs):
    config = {'num_train_timesteps': 256, 'sigma_min': 0.002, 'sigma_max': 80.0
        }
    config.update(**kwargs)
    return config
