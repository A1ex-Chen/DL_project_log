def multitask_loss(target, logits, criterion, reduce='mean'):
    """Compute multitask loss"""
    losses = {}
    for task, label in target.items():
        losses[task] = criterion(logits[task], label)
    if reduce:
        total = 0
        for _, value in losses.items():
            total += value
        if reduce == 'mean':
            losses = total / len(losses)
        elif reduce == 'sum':
            losses = total
        else:
            raise ValueError('Reduced loss must use either `mean` or `sum`!')
    return losses
