def clr_callback(mode=None, base_lr=0.0001, max_lr=0.001, gamma=0.999994):
    """Creates keras callback for cyclical learning rate."""
    if mode == 'trng1':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular')
    elif mode == 'trng2':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular2')
    elif mode == 'exp':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='exp_range',
            gamma=gamma)
    return clr
