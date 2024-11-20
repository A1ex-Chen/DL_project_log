def get_loss_fn(metric: str, reduction: str='mean'):
    if metric == LossFn.METRIC_L1:
        criterion = torch.nn.L1Loss(reduction=reduction)
    elif metric == LossFn.METRIC_L2:
        criterion = torch.nn.MSELoss(reduction=reduction)
    elif metric == LossFn.METRIC_L2_PER_ENTRY:
        if reduction == 'mean':
            criterion = lambda x, y: torch.nn.MSELoss(reduction='none')(x, y
                ).mean(list(range(x.dim()))[1:])
        elif reduction == 'sum':
            criterion = lambda x, y: torch.nn.MSELoss(reduction='none')(x, y
                ).sum(list(range(x.dim()))[1:])
        else:
            criterion = lambda x, y: torch.nn.MSELoss(reduction='none')(x, y)
    else:
        raise ValueError(f"Arguement metric doesn't support {metric}")
    return criterion
