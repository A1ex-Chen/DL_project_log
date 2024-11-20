def build_lr_scheduler(cfg: CfgNode, optimizer: torch.optim.Optimizer
    ) ->torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == 'WarmupMultiStepLR':
        steps = [x for x in cfg.SOLVER.STEPS if x <= cfg.SOLVER.MAX_ITER]
        if len(steps) != len(cfg.SOLVER.STEPS):
            logger = logging.getLogger(__name__)
            logger.warning(
                'SOLVER.STEPS contains values larger than SOLVER.MAX_ITER. These values will be ignored.'
                )
        sched = MultiStepParamScheduler(values=[(cfg.SOLVER.GAMMA ** k) for
            k in range(len(steps) + 1)], milestones=steps, num_updates=cfg.
            SOLVER.MAX_ITER)
    elif name == 'WarmupCosineLR':
        end_value = cfg.SOLVER.BASE_LR_END / cfg.SOLVER.BASE_LR
        assert end_value >= 0.0 and end_value <= 1.0, end_value
        sched = CosineParamScheduler(1, end_value)
    else:
        raise ValueError('Unknown LR scheduler: {}'.format(name))
    sched = WarmupParamScheduler(sched, cfg.SOLVER.WARMUP_FACTOR, min(cfg.
        SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0), cfg.SOLVER.
        WARMUP_METHOD)
    return LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.
        MAX_ITER)
