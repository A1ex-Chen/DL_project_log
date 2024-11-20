def main(cfg):
    dataset = build_dataset(cfg)
    detector_model = build_detector(cfg)
    load_pretrained_weights(detector_model, dataset, cfg)
    optimizer = build_optimizer(cfg, keras=True)
    optimizer = dist.DistributedOptimizer(optimizer)
    detector_model.compile(optimizer=optimizer)
    steps_per_epoch = cfg.SOLVER.NUM_IMAGES // cfg.INPUT.TRAIN_BATCH_SIZE
    epochs = cfg.SOLVER.MAX_ITERS // steps_per_epoch + 1
    callbacks = [dist.callbacks.BroadcastGlobalVariablesCallback(0)]
    detector_model.fit(x=dataset, steps_per_epoch=steps_per_epoch, epochs=
        epochs, callbacks=callbacks, verbose=1 if rank == 0 else 0)
