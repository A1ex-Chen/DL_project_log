def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length
