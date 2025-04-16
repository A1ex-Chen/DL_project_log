def build_optimizer(cfg, model):
    name = cfg.OPTIMIZER.TYPE
    if name == '':
        warnings.warn('No optimizer is built.')
        return None
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(model.parameters(), lr=cfg.
            OPTIMIZER.BASE_LR, weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY, **
            cfg.OPTIMIZER.get(name, dict()))
    else:
        raise ValueError('Unsupported type of optimizer.')
