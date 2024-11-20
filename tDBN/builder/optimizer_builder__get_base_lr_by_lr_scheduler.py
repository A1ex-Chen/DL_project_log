def _get_base_lr_by_lr_scheduler(learning_rate_config):
    base_lr = None
    learning_rate_type = learning_rate_config.WhichOneof('learning_rate')
    if learning_rate_type == 'constant_learning_rate':
        config = learning_rate_config.constant_learning_rate
        base_lr = config.learning_rate
    if learning_rate_type == 'exponential_decay_learning_rate':
        config = learning_rate_config.exponential_decay_learning_rate
        base_lr = config.initial_learning_rate
    if learning_rate_type == 'manual_step_learning_rate':
        config = learning_rate_config.manual_step_learning_rate
        base_lr = config.initial_learning_rate
        if not config.schedule:
            raise ValueError('Empty learning rate schedule.')
    if learning_rate_type == 'cosine_decay_learning_rate':
        config = learning_rate_config.cosine_decay_learning_rate
        base_lr = config.learning_rate_base
    if base_lr is None:
        raise ValueError('Learning_rate %s not supported.' % learning_rate_type
            )
    return base_lr
