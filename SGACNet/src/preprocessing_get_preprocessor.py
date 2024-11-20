def get_preprocessor(depth_mean, depth_std, depth_mode='refined', height=
    None, width=None, phase='train', train_random_rescale=(1.0, 1.4)):
    assert phase in ['train', 'test']
    if phase == 'train':
        transform_list = [RandomRescale(train_random_rescale), RandomCrop(
            crop_height=height, crop_width=width), RandomHSV((0.9, 1.1), (
            0.9, 1.1), (25, 25)), RandomFlip(), ToTensor(), Normalize(
            depth_mean=depth_mean, depth_std=depth_std, depth_mode=
            depth_mode), MultiScaleLabel(downsampling_rates=[8, 16, 32])]
    else:
        if height is None and width is None:
            transform_list = []
        else:
            transform_list = [Rescale(height=height, width=width)]
        transform_list.extend([ToTensor(), Normalize(depth_mean=depth_mean,
            depth_std=depth_std, depth_mode=depth_mode)])
    transform = transforms.Compose(transform_list)
    return transform
