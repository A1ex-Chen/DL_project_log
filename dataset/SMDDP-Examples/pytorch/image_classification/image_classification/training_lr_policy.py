def lr_policy(lr_fn, logger=None):
    if logger is not None:
        logger.register_metric('lr', log.LR_METER(), verbosity=dllogger.
            Verbosity.VERBOSE)

    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        if logger is not None:
            logger.log_metric('lr', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return _alr
