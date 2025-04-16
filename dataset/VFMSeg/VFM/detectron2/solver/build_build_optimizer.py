def build_optimizer(cfg: CfgNode, model: torch.nn.Module
    ) ->torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params = get_default_optimizer_params(model, base_lr=cfg.SOLVER.BASE_LR,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM, bias_lr_factor=cfg.
        SOLVER.BIAS_LR_FACTOR, weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS)
    return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(params, lr=cfg
        .SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, nesterov=cfg.SOLVER.
        NESTEROV, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
