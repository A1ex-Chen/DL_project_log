def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
    return TF.normalize(x, mean, std, inplace=inplace)
