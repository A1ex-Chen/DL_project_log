def _alr(optimizer, iteration, epoch):
    lr = lr_fn(iteration, epoch)
    if logger is not None:
        logger.log_metric('lr', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
