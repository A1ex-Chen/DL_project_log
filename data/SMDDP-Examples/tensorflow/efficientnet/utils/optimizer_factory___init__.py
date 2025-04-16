def __init__(self, optimizer: tf.keras.optimizers.Optimizer, average_decay:
    float=0.99, start_step: int=0, dynamic_decay: bool=True, name: Text=
    'moving_average', **kwargs):
    """Construct a new MovingAverage optimizer.

    Args:
      optimizer: `tf.keras.optimizers.Optimizer` that will be
        used to compute and apply gradients.
      average_decay: float. Decay to use to maintain the moving averages
        of trained variables.
      start_step: int. What step to start the moving average.
      dynamic_decay: bool. Whether to change the decay based on the number
        of optimizer updates. Decay will start at 0.1 and gradually increase
        up to `average_decay` after each optimizer update. This behavior is
        similar to `tf.train.ExponentialMovingAverage` in TF 1.x.
      name: Optional name for the operations created when applying
        gradients. Defaults to "moving_average".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`,
        `clipvalue`, `lr`, `decay`}.
    """
    super(MovingAverage, self).__init__(name, **kwargs)
    self._optimizer = optimizer
    self._average_decay = average_decay
    self._start_step = tf.constant(start_step, tf.float32)
    self._dynamic_decay = dynamic_decay
