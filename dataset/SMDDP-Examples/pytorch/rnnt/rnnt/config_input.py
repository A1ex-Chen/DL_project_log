def input(conf_yaml, split='train'):
    conf = copy.deepcopy(conf_yaml[f'input_{split}'])
    conf_dataset = conf.pop('audio_dataset')
    conf_features = conf.pop('filterbank_features')
    conf_splicing = conf.pop('frame_splicing', {})
    conf_padalign = conf.pop('pad_align', {})
    conf_specaugm = conf.pop('spec_augment', None)
    conf_cutoutau = conf.pop('cutout_augment', None)
    inner_classes = [(conf_dataset, 'speed_perturbation',
        SpeedPerturbationParams)]
    amp = ['optim_level']
    for conf_tgt, key, klass in inner_classes:
        if key in conf_tgt:
            conf_tgt[key] = validate_and_fill(klass, conf_tgt[key],
                optional=amp)
    for k in conf:
        raise ValueError(f'Unknown key {k}')
    conf_dataset = validate_and_fill(PipelineParams, conf_dataset)
    conf_features = validate_and_fill(features.FilterbankFeatures,
        conf_features, optional=amp)
    conf_splicing = validate_and_fill(features.FrameSplicing, conf_splicing,
        optional=amp)
    conf_padalign = validate_and_fill(features.PadAlign, conf_padalign,
        optional=amp)
    conf_specaugm = conf_specaugm and validate_and_fill(features.
        SpecAugment, conf_specaugm, optional=amp)
    for shared in ['sample_rate']:
        assert conf_dataset[shared] == conf_features[shared
            ], f'{shared} should match in Dataset and FeatureProcessor: {conf_dataset[shared]}, {conf_features[shared]}'
    return (conf_dataset, conf_features, conf_splicing, conf_padalign,
        conf_specaugm)
