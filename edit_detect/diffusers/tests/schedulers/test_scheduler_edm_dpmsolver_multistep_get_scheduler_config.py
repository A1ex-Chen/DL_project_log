def get_scheduler_config(self, **kwargs):
    config = {'sigma_min': 0.002, 'sigma_max': 80.0, 'sigma_data': 0.5,
        'num_train_timesteps': 1000, 'solver_order': 2, 'prediction_type':
        'epsilon', 'thresholding': False, 'sample_max_value': 1.0,
        'algorithm_type': 'dpmsolver++', 'solver_type': 'midpoint',
        'lower_order_final': False, 'euler_at_final': False,
        'final_sigmas_type': 'sigma_min'}
    config.update(**kwargs)
    return config
