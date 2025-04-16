def classify_transforms(size=224):
    assert isinstance(size, int
        ), f'ERROR: classify_transforms size {size} must be integer, not (list, tuple)'
    return T.Compose([CenterCrop(size), ToTensor(), T.Normalize(
        IMAGENET_MEAN, IMAGENET_STD)])
