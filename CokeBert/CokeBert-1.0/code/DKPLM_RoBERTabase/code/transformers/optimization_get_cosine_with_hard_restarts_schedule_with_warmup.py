def get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
    num_warmup_steps, num_training_steps, num_cycles=1.0, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function with several hard restarts, after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, 
            num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (float(num_cycles) *
            progress % 1.0))))
    return LambdaLR(optimizer, lr_lambda, last_epoch)
