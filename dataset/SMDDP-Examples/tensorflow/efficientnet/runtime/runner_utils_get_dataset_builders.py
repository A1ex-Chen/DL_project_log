def get_dataset_builders(params, one_hot):
    """Create and return train and validation dataset builders."""
    if sdp.size() > 1:
        num_gpus = sdp.size()
    else:
        num_devices = 1
    image_size = get_image_size_from_model(params.arch)
    print('Image size {}'.format(image_size))
    print('Train batch size {}'.format(params.train_batch_size))
    builders = []
    validation_dataset_builder = None
    train_dataset_builder = None
    if 'train' in params.mode:
        train_dataset_builder = dataset_factory.Dataset(data_dir=params.
            data_dir, index_file_dir=params.index_file, split='train',
            num_classes=params.num_classes, image_size=image_size,
            batch_size=params.train_batch_size, one_hot=one_hot, use_dali=
            params.use_dali, augmenter=params.augmenter_name,
            augmenter_params=build_augmenter_params(params.augmenter_name,
            params.cutout_const, params.translate_const, params.num_layers,
            params.magnitude, params.autoaugmentation_name), mixup_alpha=
            params.mixup_alpha)
    if 'eval' in params.mode:
        validation_dataset_builder = dataset_factory.Dataset(data_dir=
            params.data_dir, index_file_dir=params.index_file, split=
            'validation', num_classes=params.num_classes, image_size=
            image_size, batch_size=params.eval_batch_size, one_hot=one_hot,
            use_dali=params.use_dali_eval)
    builders.append(train_dataset_builder)
    builders.append(validation_dataset_builder)
    return builders
