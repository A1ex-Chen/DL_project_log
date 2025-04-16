def build_compute_metric(compute_metric_cfg, preprocessor):
    if compute_metric_cfg is not None:
        compute_metric_cfg = dict(compute_metric_cfg)
        compute_metric_cfg.update(dict(preprocessor=preprocessor))
        compute_metrics = METRICS.build(cfg=compute_metric_cfg)
    else:
        compute_metrics = None
    return compute_metrics
