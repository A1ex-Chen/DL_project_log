def create_optimizer(init_lr: float, num_train_steps: int, num_warmup_steps:
    int, min_lr_ratio: float=0.0, adam_beta1: float=0.9, adam_beta2: float=
    0.999, adam_epsilon: float=1e-08, weight_decay_rate: float=0.0, power:
    float=1.0, include_in_weight_decay: Optional[List[str]]=None):
    """
    Creates an optimizer with a learning rate schedule using a warmup phase followed by a linear decay.

    Args:
        init_lr (:obj:`float`):
            The desired learning rate at the end of the warmup phase.
        num_train_steps (:obj:`int`):
            The total number of training steps.
        num_warmup_steps (:obj:`int`):
            The number of warmup steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0):
            The final learning rate at the end of the linear decay will be :obj:`init_lr * min_lr_ratio`.
        adam_beta1 (:obj:`float`, `optional`, defaults to 0.9):
            The beta1 to use in Adam.
        adam_beta2 (:obj:`float`, `optional`, defaults to 0.999):
            The beta2 to use in Adam.
        adam_epsilon (:obj:`float`, `optional`, defaults to 1e-8):
            The epsilon to use in Adam.
        weight_decay_rate (:obj:`float`, `optional`, defaults to 0):
            The weight decay to use.
        power (:obj:`float`, `optional`, defaults to 1.0):
            The power to use for PolynomialDecay.
        include_in_weight_decay (:obj:`List[str]`, `optional`):
            List of the parameter names (or re patterns) to apply weight decay to. If none is passed, weight decay is
            applied to all parameters except bias and layer norm parameters.
    """
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=init_lr, decay_steps=num_train_steps -
        num_warmup_steps, end_learning_rate=init_lr * min_lr_ratio, power=power
        )
    if num_warmup_steps:
        lr_schedule = WarmUp(initial_learning_rate=init_lr,
            decay_schedule_fn=lr_schedule, warmup_steps=num_warmup_steps)
    if weight_decay_rate > 0.0:
        optimizer = AdamWeightDecay(learning_rate=lr_schedule,
            weight_decay_rate=weight_decay_rate, beta_1=adam_beta1, beta_2=
            adam_beta2, epsilon=adam_epsilon, exclude_from_weight_decay=[
            'LayerNorm', 'layer_norm', 'bias'], include_in_weight_decay=
            include_in_weight_decay)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,
            beta_1=adam_beta1, beta_2=adam_beta2, epsilon=adam_epsilon)
    return optimizer, lr_schedule
