def posterize_func(img, bits):
    """
    same output as PIL.ImageOps.posterize
    """
    out = np.bitwise_and(img, np.uint8(255 << 8 - bits))
    return out
