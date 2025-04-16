def get_distortion_transforms(distortion_name, img_size, severity=1, **kwargs):
    distortion_transform = transforms.Lambda(generate_distortion_fn(
        distortion_name, severity=severity))
    train_transforms, val_transforms = get_vanilla_transforms(img_size,
        add_train_transforms=distortion_transform, add_test_transforms=
        distortion_transform, **kwargs)
    return train_transforms, val_transforms
