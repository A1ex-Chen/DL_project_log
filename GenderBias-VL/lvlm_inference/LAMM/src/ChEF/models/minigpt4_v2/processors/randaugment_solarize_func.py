def solarize_func(img, thresh=128):
    """
    same output as PIL.ImageOps.posterize
    """
    table = np.array([(el if el < thresh else 255 - el) for el in range(256)])
    table = table.clip(0, 255).astype(np.uint8)
    out = table[img]
    return out
