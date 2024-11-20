def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-05):
    g = [], [], []
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g[2].append(v.bias)
        if isinstance(v, bn):
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g[0].append(v.weight)
    if name == 'Adam':
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999),
            weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum,
            nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')
    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups {len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias"
        )
    return optimizer
