def load_pretrained_weights(detector_model, dataset, cfg):
    print('Loading checkpoint')
    features, labels = next(iter(dataset))
    _ = detector_model(features, training=False)
    chkp = tf.compat.v1.train.NewCheckpointReader(cfg.PATHS.WEIGHTS)
    weights = [chkp.get_tensor(i) for i in ['/'.join(i.name.split('/')[-2:]
        ).split(':')[0] for i in detector_model.layers[0].weights]]
    detector_model.layers[0].set_weights(weights)
