def main(cfg):
    tf.config.optimizer.set_experimental_options({'auto_mixed_precision':
        cfg.SOLVER.FP16})
    tf.config.optimizer.set_jit(cfg.SOLVER.XLA)
    if int(tf.__version__.split('.')[1]) >= 4:
        tf.config.experimental.enable_tensor_float_32_execution(cfg.SOLVER.TF32
            )
    dataset = iter(build_dataset(cfg))
    detector = build_detector(cfg)
    features, labels = next(dataset)
    result = detector(features, training=False)
    optimizer = build_optimizer(cfg)
    trainer = build_trainer(cfg, detector, optimizer, dist='smd' if
        is_sm_dist() else 'hvd')
    runner = Runner(trainer, cfg)
    hooks = build_hooks(cfg)
    for hook in hooks:
        runner.register_hook(hook)
    runner.run(dataset)
