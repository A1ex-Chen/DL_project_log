def lr_cosine_policy(base_lr, warmup_length, epochs, end_lr=0, logger=None):

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = end_lr + 0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr -
                end_lr)
        return lr
    return lr_policy(_lr_fn, logger=logger)
