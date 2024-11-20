def get_constant_schedule_with_warmup(optimizer, num_warmup_steps,
    last_epoch=-1):
    """ Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
