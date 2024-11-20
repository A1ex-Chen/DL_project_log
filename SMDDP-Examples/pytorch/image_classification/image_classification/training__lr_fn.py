def _lr_fn(iteration, epoch):
    if epoch < warmup_length:
        lr = base_lr * (epoch + 1) / warmup_length
    else:
        e = epoch - warmup_length
        lr = base_lr * epoch_decay ** math.floor(e / decay_step)
    return lr
