def _create_learning_rate_scheduler(learning_rate_config, optimizer,
    last_step=-1):
    """Create optimizer learning rate scheduler based on config.

  Args:
    learning_rate_config: A LearningRate proto message.

  Returns:
    A learning rate.

  Raises:
    ValueError: when using an unsupported input data type.
  """
    lr_scheduler = None
    learning_rate_type = learning_rate_config.WhichOneof('learning_rate')
    if learning_rate_type == 'constant_learning_rate':
        config = learning_rate_config.constant_learning_rate
        lr_scheduler = learning_schedules.Constant(optimizer, last_step=
            last_step)
    if learning_rate_type == 'exponential_decay_learning_rate':
        config = learning_rate_config.exponential_decay_learning_rate
        lr_scheduler = learning_schedules.ExponentialDecay(optimizer,
            config.decay_steps, config.decay_factor, config.staircase,
            last_step=last_step)
    if learning_rate_type == 'manual_step_learning_rate':
        config = learning_rate_config.manual_step_learning_rate
        if not config.schedule:
            raise ValueError('Empty learning rate schedule.')
        learning_rate_step_boundaries = [x.step for x in config.schedule]
        learning_rate_sequence = [config.initial_learning_rate]
        learning_rate_sequence += [x.learning_rate for x in config.schedule]
        lr_scheduler = learning_schedules.ManualStepping(optimizer,
            learning_rate_step_boundaries, learning_rate_sequence,
            last_step=last_step)
    if learning_rate_type == 'cosine_decay_learning_rate':
        config = learning_rate_config.cosine_decay_learning_rate
        lr_scheduler = learning_schedules.CosineDecayWithWarmup(optimizer,
            config.total_steps, config.warmup_learning_rate, config.
            warmup_steps, last_step=last_step)
    if lr_scheduler is None:
        raise ValueError('Learning_rate %s not supported.' % learning_rate_type
            )
    return lr_scheduler
