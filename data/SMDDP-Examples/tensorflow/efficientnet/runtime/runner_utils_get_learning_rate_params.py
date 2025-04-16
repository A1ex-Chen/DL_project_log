def get_learning_rate_params(name, initial_lr, decay_epochs, decay_rate,
    warmup_epochs):
    return {'name': name, 'initial_lr': initial_lr, 'decay_epochs':
        decay_epochs, 'decay_rate': decay_rate, 'warmup_epochs':
        warmup_epochs, 'examples_per_epoch': None, 'boundaries': None,
        'multipliers': None, 'scale_by_batch_size': 1.0 / 128.0,
        'staircase': True}
