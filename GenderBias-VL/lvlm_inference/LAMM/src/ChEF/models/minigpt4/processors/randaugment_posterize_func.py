def posterize_func(img, bits):
    """
    same output as PIL.ImageOps.posterize
    """
    out = np.bitwise_and(img, np.uint8(255 << 8 - bits))
    return out
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, 0], [factor, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.
        INTER_LINEAR).astype(np.uint8)
    return out
