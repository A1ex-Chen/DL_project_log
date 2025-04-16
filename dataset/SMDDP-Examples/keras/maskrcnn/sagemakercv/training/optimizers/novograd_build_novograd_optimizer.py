@OPTIMIZERS.register('NovoGrad')
def build_novograd_optimizer(cfg, scheduler):
    return NovoGrad(learning_rate=scheduler, beta_1=cfg.SOLVER.BETA_1,
        beta_2=cfg.SOLVER.BETA_2, epsilon=cfg.SOLVER.EPSILON, weight_decay=
        cfg.SOLVER.WEIGHT_DECAY)
