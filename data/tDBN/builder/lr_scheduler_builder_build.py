def build(optimizer_config, optimizer, last_step=-1):
    """Create lr scheduler based on config. note that
  lr_scheduler must accept a optimizer that has been restored.

  Args:
    optimizer_config: A Optimizer proto message.

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
    optimizer_type = optimizer_config.WhichOneof('optimizer')
    if optimizer_type == 'rms_prop_optimizer':
        config = optimizer_config.rms_prop_optimizer
        lr_scheduler = _create_learning_rate_scheduler(config.learning_rate,
            optimizer, last_step=last_step)
    if optimizer_type == 'momentum_optimizer':
        config = optimizer_config.momentum_optimizer
        lr_scheduler = _create_learning_rate_scheduler(config.learning_rate,
            optimizer, last_step=last_step)
    if optimizer_type == 'adam_optimizer':
        config = optimizer_config.adam_optimizer
        lr_scheduler = _create_learning_rate_scheduler(config.learning_rate,
            optimizer, last_step=last_step)
    return lr_scheduler
