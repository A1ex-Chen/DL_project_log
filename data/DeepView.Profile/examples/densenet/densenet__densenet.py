def _densenet(arch, growth_rate, block_config, num_init_features,
    pretrained, progress, **kwargs):
    return DenseNet(growth_rate, block_config, num_init_features, **kwargs)
