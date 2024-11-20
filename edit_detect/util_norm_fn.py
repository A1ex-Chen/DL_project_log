@staticmethod
def norm_fn(loss_type: str):
    reduction = 'none'
    if loss_type == DistanceFn.NORM_L1:
        return partial(F.l1_loss, reduction=reduction)
    elif loss_type == DistanceFn.NORM_L2:
        return partial(F.mse_loss, reduction=reduction)
    elif loss_type == DistanceFn.NORM_HUBER:
        return partial(F.smooth_l1_loss, reduction=reduction)
    else:
        raise NotImplementedError()
