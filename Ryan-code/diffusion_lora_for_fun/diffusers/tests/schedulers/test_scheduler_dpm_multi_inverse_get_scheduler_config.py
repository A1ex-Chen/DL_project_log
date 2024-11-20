def get_scheduler_config(self, **kwargs):
    config = {'num_train_timesteps': 1000, 'beta_start': 0.0001, 'beta_end':
        0.02, 'beta_schedule': 'linear', 'solver_order': 2,
        'prediction_type': 'epsilon', 'thresholding': False,
        'sample_max_value': 1.0, 'algorithm_type': 'dpmsolver++',
        'solver_type': 'midpoint', 'lower_order_final': False,
        'lambda_min_clipped': -float('inf'), 'variance_type': None}
    config.update(**kwargs)
    return config
