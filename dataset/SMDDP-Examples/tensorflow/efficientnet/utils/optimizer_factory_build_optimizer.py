def build_optimizer(optimizer_name: Text, base_learning_rate: tf.keras.
    optimizers.schedules.LearningRateSchedule, params: Dict[Text, Any]):
    """Build the optimizer based on name.

  Args:
    optimizer_name: String representation of the optimizer name. Examples:
      sgd, momentum, rmsprop.
    base_learning_rate: `tf.keras.optimizers.schedules.LearningRateSchedule`
      base learning rate.
    params: String -> Any dictionary representing the optimizer params.
      This should contain optimizer specific parameters such as
      `base_learning_rate`, `decay`, etc.

  Returns:
    A tf.keras.Optimizer.

  Raises:
    ValueError if the provided optimizer_name is not supported.

  """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'sgd':
        nesterov = params.get('nesterov', False)
        optimizer = tf.keras.optimizers.SGD(learning_rate=
            base_learning_rate, nesterov=nesterov)
    elif optimizer_name == 'momentum':
        nesterov = params.get('nesterov', False)
        optimizer = tf.keras.optimizers.SGD(learning_rate=
            base_learning_rate, momentum=params['momentum'], nesterov=nesterov)
    elif optimizer_name == 'rmsprop':
        rho = params.get('decay', None) or params.get('rho', 0.9)
        momentum = params.get('momentum', 0.9)
        epsilon = params.get('epsilon', 1e-07)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=
            base_learning_rate, rho=rho, momentum=momentum, epsilon=epsilon)
    elif optimizer_name == 'adam':
        beta_1 = params.get('beta_1', 0.9)
        beta_2 = params.get('beta_2', 0.999)
        epsilon = params.get('epsilon', 1e-07)
        optimizer = tf.keras.optimizers.Adam(learning_rate=
            base_learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    elif optimizer_name == 'adamw':
        weight_decay = params.get('weight_decay', 0.01)
        beta_1 = params.get('beta_1', 0.9)
        beta_2 = params.get('beta_2', 0.999)
        epsilon = params.get('epsilon', 1e-07)
        optimizer = tfa.optimizers.AdamW(weight_decay=weight_decay,
            learning_rate=base_learning_rate, beta_1=beta_1, beta_2=beta_2,
            epsilon=epsilon)
    else:
        raise ValueError('Unknown optimizer %s' % optimizer_name)
    if params.get('lookahead', None):
        optimizer = tfa.optimizers.Lookahead(optimizer)
    moving_average_decay = params.get('moving_average_decay', 0.0)
    if moving_average_decay is not None and moving_average_decay > 0.0:
        optimizer = MovingAverage(optimizer, average_decay=moving_average_decay
            )
    return optimizer
