def contrast_func(img, factor):
    """
        same output as PIL.ImageEnhance.Contrast
    """
    mean = np.sum(np.mean(img, axis=(0, 1)) * np.array([0.114, 0.587, 0.299]))
    table = np.array([((el - mean) * factor + mean) for el in range(256)]
        ).clip(0, 255).astype(np.uint8)
    out = table[img]
    return out
