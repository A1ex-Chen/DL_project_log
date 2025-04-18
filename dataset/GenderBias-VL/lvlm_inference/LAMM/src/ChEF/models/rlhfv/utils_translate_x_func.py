def translate_x_func(img, offset, fill=(0, 0, 0)):
    """
        same output as PIL.Image.transform
    """
    H, W = img.shape[0], img.shape[1]
    M = np.float32([[1, 0, -offset], [0, 1, 0]])
    out = cv2.warpAffine(img, M, (W, H), borderValue=fill, flags=cv2.
        INTER_LINEAR).astype(np.uint8)
    return out
