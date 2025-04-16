def lr_lambda(current_step: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step > num_training_steps:
        return lr_end / lr_init
    else:
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining ** power + lr_end
        return decay / lr_init
