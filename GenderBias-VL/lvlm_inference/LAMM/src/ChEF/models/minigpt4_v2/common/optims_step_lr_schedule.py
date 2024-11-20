def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * decay_rate ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
