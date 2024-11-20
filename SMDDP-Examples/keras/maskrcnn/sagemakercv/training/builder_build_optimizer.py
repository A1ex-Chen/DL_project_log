def build_optimizer(cfg, keras=False):
    scheduler = build_scheduler(cfg, keras)
    optimizer = OPTIMIZERS[cfg.SOLVER.OPTIMIZER](cfg, scheduler)
    if cfg.SOLVER.FP16 and not keras:
        if int(tf.__version__.split('.')[1]) < 4:
            optimizer = (tf.keras.mixed_precision.experimental.
                LossScaleOptimizer(optimizer, 'dynamic'))
        else:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer,
                dynamic=True, initial_scale=2 ** 15, dynamic_growth_steps=2000)
    return optimizer
