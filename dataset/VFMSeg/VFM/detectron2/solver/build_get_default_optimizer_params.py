def get_default_optimizer_params(model: torch.nn.Module, base_lr: Optional[
    float]=None, weight_decay: Optional[float]=None, weight_decay_norm:
    Optional[float]=None, bias_lr_factor: Optional[float]=1.0,
    weight_decay_bias: Optional[float]=None, overrides: Optional[Dict[str,
    Dict[str, float]]]=None) ->List[Dict[str, Any]]:
    """
    Get default param list for optimizer, with support for a few types of
    overrides. If no overrides needed, this is equivalent to `model.parameters()`.

    Args:
        base_lr: lr for every group by default. Can be omitted to use the one in optimizer.
        weight_decay: weight decay for every group by default. Can be omitted to use the one
            in optimizer.
        weight_decay_norm: override weight decay for params in normalization layers
        bias_lr_factor: multiplier of lr for bias parameters.
        weight_decay_bias: override weight decay for bias parameters
        overrides: if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            ``{"embedding": {"lr": 0.01, "weight_decay": 0.1}}`` will set the LR and
            weight decay values for all module parameters named `embedding`.

    For common detection models, ``weight_decay_norm`` is the only option
    needed to be set. ``bias_lr_factor,weight_decay_bias`` are legacy settings
    from Detectron1 that are not found useful.

    Example:
    ::
        torch.optim.SGD(get_default_optimizer_params(model, weight_decay_norm=0),
                       lr=0.01, weight_decay=1e-4, momentum=0.9)
    """
    if overrides is None:
        overrides = {}
    defaults = {}
    if base_lr is not None:
        defaults['lr'] = base_lr
    if weight_decay is not None:
        defaults['weight_decay'] = weight_decay
    bias_overrides = {}
    if bias_lr_factor is not None and bias_lr_factor != 1.0:
        if base_lr is None:
            raise ValueError('bias_lr_factor requires base_lr')
        bias_overrides['lr'] = base_lr * bias_lr_factor
    if weight_decay_bias is not None:
        bias_overrides['weight_decay'] = weight_decay_bias
    if len(bias_overrides):
        if 'bias' in overrides:
            raise ValueError("Conflicting overrides for 'bias'")
        overrides['bias'] = bias_overrides
    norm_module_types = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.
        nn.BatchNorm3d, torch.nn.SyncBatchNorm, torch.nn.GroupNorm, torch.
        nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm, torch.nn.LocalResponseNorm)
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module in model.modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            if value in memo:
                continue
            memo.add(value)
            hyperparams = copy.copy(defaults)
            if isinstance(module, norm_module_types
                ) and weight_decay_norm is not None:
                hyperparams['weight_decay'] = weight_decay_norm
            hyperparams.update(overrides.get(module_param_name, {}))
            params.append({'params': [value], **hyperparams})
    return reduce_param_groups(params)
