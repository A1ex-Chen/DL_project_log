@HOOKS.register('TensorboardMetricsLogger')
def build_tensorboard_metrics_logger(cfg):
    return TensorboardMetricsLogger(interval=cfg.LOG_INTERVAL)
