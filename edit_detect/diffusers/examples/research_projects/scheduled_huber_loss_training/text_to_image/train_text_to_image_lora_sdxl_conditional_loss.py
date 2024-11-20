def conditional_loss(model_pred: torch.Tensor, target: torch.Tensor,
    reduction: str='mean', loss_type: str='l2', huber_c: float=0.1):
    if loss_type == 'l2':
        loss = F.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == 'huber' or loss_type == 'huber_scheduled':
        loss = huber_c * (torch.sqrt((model_pred - target) ** 2 + huber_c **
            2) - huber_c)
        if reduction == 'mean':
            loss = torch.mean(loss)
        elif reduction == 'sum':
            loss = torch.sum(loss)
    else:
        raise NotImplementedError(f'Unsupported Loss Type {loss_type}')
    return loss
