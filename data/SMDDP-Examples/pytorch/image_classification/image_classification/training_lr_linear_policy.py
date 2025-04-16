def lr_linear_policy(base_lr, warmup_length, epochs, logger=None):

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = base_lr * (1 - e / es)
        return lr
    return lr_policy(_lr_fn, logger=logger)
