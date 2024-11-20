def master_params(optimizer):
    """
    Generator expression that iterates over the params owned by ``optimizer``.

    Args:
        optimizer: An optimizer previously returned from ``amp.initialize``.
    """
    for group in optimizer.param_groups:
        for p in group['params']:
            yield p
