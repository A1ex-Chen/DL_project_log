def get_lr(optim, lr_sched=None):
    if optim is not None:
        return optim.param_groups[0]['lr']
    elif lr_sched is not None:
        return lr_sched.get_last_lr()[0]
    else:
        raise ValueError(
            f'Arguement optim and lr_sched should not be None in the same time'
            )
