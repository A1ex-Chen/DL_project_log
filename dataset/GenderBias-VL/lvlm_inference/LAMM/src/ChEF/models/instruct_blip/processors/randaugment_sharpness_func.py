def sharpness_func(img, factor):
    """
    The differences the this result and PIL are all on the 4 boundaries, the center
    areas are same
    """
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1][1] = 5
    kernel /= 13
    degenerate = cv2.filter2D(img, -1, kernel)
    if factor == 0.0:
        out = degenerate
    elif factor == 1.0:
        out = img
    else:
        out = img.astype(np.float32)
        degenerate = degenerate.astype(np.float32)[1:-1, 1:-1, :]
        out[1:-1, 1:-1, :] = degenerate + factor * (out[1:-1, 1:-1, :] -
            degenerate)
        out = out.astype(np.uint8)
    return out
