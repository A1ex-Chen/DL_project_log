def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x
