def keras_default_config():
    """Defines parameters that intervine in different functions using the keras defaults.
    This helps to keep consistency in parameters between frameworks.
    """
    kerasDefaults = {}
    kerasDefaults['decay_lr'] = 0.0
    kerasDefaults['epsilon'] = 1e-08
    kerasDefaults['rho'] = 0.9
    kerasDefaults['momentum_sgd'] = 0.0
    kerasDefaults['nesterov_sgd'] = False
    kerasDefaults['beta_1'] = 0.9
    kerasDefaults['beta_2'] = 0.999
    kerasDefaults['decay_schedule_lr'] = 0.004
    kerasDefaults['minval_uniform'] = -0.05
    kerasDefaults['maxval_uniform'] = 0.05
    kerasDefaults['mean_normal'] = 0.0
    kerasDefaults['stddev_normal'] = 0.05
    return kerasDefaults
