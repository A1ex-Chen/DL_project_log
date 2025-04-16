def build_scheduler(cfg, keras=False):
    scheduler = SCHEDULERS[cfg.SOLVER.SCHEDULE](cfg)
    if cfg.SOLVER.WARMUP and not keras:
        scheduler = SCHEDULERS[cfg.SOLVER.WARMUP](cfg, scheduler)
    return scheduler
