def build_learning_rate(params: Dict[Text, Any], batch_size: int=None,
    train_steps: int=None, max_epochs: int=None):
    """Build the learning rate given the provided configuration."""
    decay_type = params['name']
    base_lr = params['initial_lr']
    decay_rate = params['decay_rate']
    if params['decay_epochs'] is not None:
        decay_steps = params['decay_epochs'] * train_steps
    else:
        decay_steps = 0
    if params['warmup_epochs'] is not None:
        warmup_steps = params['warmup_epochs'] * train_steps
    else:
        warmup_steps = 0
    lr_multiplier = params['scale_by_batch_size']
    if lr_multiplier and lr_multiplier > 0:
        base_lr *= lr_multiplier * batch_size
    if decay_type == 'exponential':
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=base_lr, decay_steps=decay_steps,
            decay_rate=decay_rate, staircase=params['staircase'])
    elif decay_type == 'piecewise_constant_with_warmup':
        lr = learning_rate.PiecewiseConstantDecayWithWarmup(batch_size=
            batch_size, epoch_size=params['examples_per_epoch'],
            warmup_epochs=params['warmup_epochs'], boundaries=params[
            'boundaries'], multipliers=params['multipliers'])
    elif decay_type == 'cosine':
        decay_steps = (max_epochs - params['warmup_epochs']) * train_steps
        lr = tf.keras.experimental.CosineDecay(initial_learning_rate=
            base_lr, decay_steps=decay_steps, alpha=0.0)
    elif decay_type == 'linearcosine':
        decay_steps = (max_epochs - params['warmup_epochs']) * train_steps
        lr = tf.keras.experimental.NoisyLinearCosineDecay(initial_learning_rate
            =base_lr, decay_steps=decay_steps, initial_variance=0.5,
            variance_decay=0.55, num_periods=0.5, alpha=0.0, beta=0.001)
    if warmup_steps > 0:
        if decay_type != 'piecewise_constant_with_warmup':
            lr = learning_rate.WarmupDecaySchedule(lr, warmup_steps)
    return lr
