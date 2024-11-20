def get_scheduler_config(self, **kwargs):
    config = {'num_vec_classes': 4097, 'num_train_timesteps': 100}
    config.update(**kwargs)
    return config
