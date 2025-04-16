def square_transform(size=224):
    return transforms.Compose([transforms.Resize((size, size),
        interpolation=transforms.InterpolationMode.BICUBIC), transforms.
        ToTensor(), inception_normalize])
