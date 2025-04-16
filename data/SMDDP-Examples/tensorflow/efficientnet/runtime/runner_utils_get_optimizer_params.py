def get_optimizer_params(name, decay, epsilon, momentum,
    moving_average_decay, nesterov, beta_1, beta_2):
    return {'name': name, 'decay': decay, 'epsilon': epsilon, 'momentum':
        momentum, 'moving_average_decay': moving_average_decay, 'nesterov':
        nesterov, 'beta_1': beta_1, 'beta_2': beta_2}
