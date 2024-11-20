def lr_step_policy(base_lr, steps, decay_factor, warmup_length, logger=None):

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr
    return lr_policy(_lr_fn, logger=logger)
