def build_trainer(cfg, model, optimizer, dist=None):
    return TRAINERS[cfg.SOLVER.TRAINER](cfg, model, optimizer, dist)
