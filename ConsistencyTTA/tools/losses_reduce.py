def reduce(loss, reduction):
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'instance':
        return loss
    else:
        raise ValueError('Unknown loss reduction option.')
