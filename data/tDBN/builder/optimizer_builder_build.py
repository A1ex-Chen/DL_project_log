def build(optimizer_config, params, name=None):
    """Create optimizer based on config.

  Args:
    optimizer_config: A Optimizer proto message.

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
    optimizer_type = optimizer_config.WhichOneof('optimizer')
    optimizer = None
    if optimizer_type == 'rms_prop_optimizer':
        config = optimizer_config.rms_prop_optimizer
        optimizer = torch.optim.RMSprop(params, lr=
            _get_base_lr_by_lr_scheduler(config.learning_rate), alpha=
            config.decay, momentum=config.momentum_optimizer_value, eps=
            config.epsilon, weight_decay=config.weight_decay)
    if optimizer_type == 'momentum_optimizer':
        config = optimizer_config.momentum_optimizer
        optimizer = torch.optim.SGD(params, lr=_get_base_lr_by_lr_scheduler
            (config.learning_rate), momentum=config.
            momentum_optimizer_value, weight_decay=config.weight_decay)
    if optimizer_type == 'adam_optimizer':
        config = optimizer_config.adam_optimizer
        optimizer = torch.optim.Adam(params, lr=
            _get_base_lr_by_lr_scheduler(config.learning_rate),
            weight_decay=config.weight_decay)
    if optimizer is None:
        raise ValueError('Optimizer %s not supported.' % optimizer_type)
    if optimizer_config.use_moving_average:
        raise ValueError("torch don't support moving average")
    if name is None:
        optimizer.name = optimizer_type
    else:
        optimizer.name = name
    return optimizer
