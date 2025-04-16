@SCHEDULERS.register('LinearWarmup')
def build_linear_warmup(cfg, scheduler):
    return LinearWarmup(scheduler=scheduler, warmup_ratio=cfg.SOLVER.
        WARM_UP_RATIO, warmup_steps=cfg.SOLVER.WARMUP_STEPS, overlap=cfg.
        SOLVER.WARMUP_OVERLAP)
