def lr_policy(step, epoch, initial_lr, optimizer, steps_per_epoch,
    warmup_epochs, hold_epochs, min_lr=1e-05, exp_gamma=None, dist_lamb=False):
    """
    learning rate decay
    Args:
        initial_lr: base learning rate
        step: current iteration number
        N: total number of iterations over which learning rate is decayed
        lr_steps: list of steps to apply exp_gamma
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    hold_steps = hold_epochs * steps_per_epoch
    assert exp_gamma is not None
    if step < warmup_steps:
        a = (step + 1) / (warmup_steps + 1)
    elif step < warmup_steps + hold_steps:
        a = 1.0
    else:
        a = exp_gamma ** (epoch - warmup_epochs - hold_epochs)
    if not dist_lamb:
        optimizer.param_groups[0]['lr'] = max(a * initial_lr, min_lr)
    else:
        optimizer._lr = max(a * initial_lr, min_lr * torch.ones_like(
            initial_lr))
