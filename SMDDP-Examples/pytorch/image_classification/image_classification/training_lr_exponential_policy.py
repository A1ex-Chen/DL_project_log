def lr_exponential_policy(base_lr, warmup_length, epochs, final_multiplier=
    0.001, decay_factor=None, decay_step=1, logger=None):
    """Exponential lr policy. Setting decay factor parameter overrides final_multiplier"""
    es = epochs - warmup_length
    if decay_factor is not None:
        epoch_decay = decay_factor
    else:
        epoch_decay = np.power(2, np.log2(final_multiplier) / math.floor(es /
            decay_step))

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            lr = base_lr * epoch_decay ** math.floor(e / decay_step)
        return lr
    return lr_policy(_lr_fn, logger=logger)
