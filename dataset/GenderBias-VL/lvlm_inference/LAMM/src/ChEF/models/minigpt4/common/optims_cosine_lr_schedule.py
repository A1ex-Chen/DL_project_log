def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * epoch /
        max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
