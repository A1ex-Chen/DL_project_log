def conditional_loss(model_pred: torch.Tensor, target: torch.Tensor,
    reduction: str='mean', loss_type: str='l2', huber_c: float=0.1,
    weighting: Optional[torch.Tensor]=None):
    if loss_type == 'l2':
        if weighting is not None:
            loss = torch.mean((weighting * (model_pred.float() - target.
                float()) ** 2).reshape(target.shape[0], -1), 1)
            if reduction == 'mean':
                loss = torch.mean(loss)
            elif reduction == 'sum':
                loss = torch.sum(loss)
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction
                =reduction)
    elif loss_type == 'huber':
        if weighting is not None:
            loss = torch.mean((2 * huber_c * (torch.sqrt(weighting.float() *
                (model_pred.float() - target.float()) ** 2 + huber_c ** 2) -
                huber_c)).reshape(target.shape[0], -1), 1)
            if reduction == 'mean':
                loss = torch.mean(loss)
            elif reduction == 'sum':
                loss = torch.sum(loss)
        else:
            loss = 2 * huber_c * (torch.sqrt((model_pred - target) ** 2 + 
                huber_c ** 2) - huber_c)
            if reduction == 'mean':
                loss = torch.mean(loss)
            elif reduction == 'sum':
                loss = torch.sum(loss)
    elif loss_type == 'smooth_l1':
        if weighting is not None:
            loss = torch.mean((2 * (torch.sqrt(weighting.float() * (
                model_pred.float() - target.float()) ** 2 + huber_c ** 2) -
                huber_c)).reshape(target.shape[0], -1), 1)
            if reduction == 'mean':
                loss = torch.mean(loss)
            elif reduction == 'sum':
                loss = torch.sum(loss)
        else:
            loss = 2 * (torch.sqrt((model_pred - target) ** 2 + huber_c ** 
                2) - huber_c)
            if reduction == 'mean':
                loss = torch.mean(loss)
            elif reduction == 'sum':
                loss = torch.sum(loss)
    else:
        raise NotImplementedError(f'Unsupported Loss Type {loss_type}')
    return loss
