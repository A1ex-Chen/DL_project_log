@OPTIMIZERS.register('Momentum')
def build_momentum_optimizer(cfg, scheduler):
    return MomentumOptimizer(learning_rate=scheduler, momentum=cfg.SOLVER.
        MOMENTUM, nesterov=cfg.SOLVER.NESTEROV)
