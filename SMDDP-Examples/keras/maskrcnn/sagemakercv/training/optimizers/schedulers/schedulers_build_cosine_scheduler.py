@SCHEDULERS.register('CosineDecay')
def build_cosine_scheduler(cfg):
    scheduler = tf.keras.experimental.CosineDecay
    return scheduler(cfg.SOLVER.LR, cfg.SOLVER.MAX_ITERS, cfg.SOLVER.ALPHA)
