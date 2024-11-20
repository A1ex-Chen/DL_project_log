@SCHEDULERS.register('PiecewiseConstantDecay')
def build_piecewise_scheduler(cfg):
    assert len(cfg.SOLVER.DECAY_STEPS) == len(cfg.SOLVER.DECAY_LR)
    scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay
    return scheduler(cfg.SOLVER.DECAY_STEPS, [cfg.SOLVER.LR] + [(cfg.SOLVER
        .LR * decay) for decay in cfg.SOLVER.DECAY_LR])
