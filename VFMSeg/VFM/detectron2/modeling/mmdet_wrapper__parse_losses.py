def _parse_losses(losses: Dict[str, Tensor]) ->Dict[str, Tensor]:
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')
        if 'loss' not in loss_name:
            storage = get_event_storage()
            value = log_vars.pop(loss_name).cpu().item()
            storage.put_scalar(loss_name, value)
    return log_vars
