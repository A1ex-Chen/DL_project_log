def brightness_func(img, factor):
    """
        same output as PIL.ImageEnhance.Contrast
    """
    table = (np.arange(256, dtype=np.float32) * factor).clip(0, 255).astype(np
        .uint8)
    out = table[img]
    return out
