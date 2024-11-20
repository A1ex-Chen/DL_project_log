def mixup(im, labels, im2, labels2):
    """Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf."""
    r = np.random.beta(32.0, 32.0)
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels
