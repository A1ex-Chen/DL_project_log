def get_scheduler_config(self, **kwargs):
    config = {'num_train_timesteps': 1000, 'beta_start': 0.0001, 'beta_end':
        0.02, 'beta_schedule': 'linear', 'solver_order': 2, 'solver_type':
        'bh2', 'final_sigmas_type': 'sigma_min'}
    config.update(**kwargs)
    return config
