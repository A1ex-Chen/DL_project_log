def build_transform(is_train, randaug=True, input_size=224, interpolation=
    'bicubic'):
    if is_train:
        t = [RandomResizedCropAndInterpolation(input_size, scale=(0.5, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC), transforms
            .RandomHorizontalFlip()]
        if randaug:
            t.append(RandomAugment(2, 7, isPIL=True, augs=['Identity',
                'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']))
        t += [transforms.ToTensor(), transforms.Normalize(mean=
            IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose([transforms.Resize((input_size, input_size),
            interpolation=transforms.InterpolationMode.BICUBIC), transforms
            .ToTensor(), transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN,
            std=IMAGENET_INCEPTION_STD)])
    return t
