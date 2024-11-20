def color_func(img, factor):
    """
    same output as PIL.ImageEnhance.Color
    """
    M = np.float32([[0.886, -0.114, -0.114], [-0.587, 0.413, -0.587], [-
        0.299, -0.299, 0.701]]) * factor + np.float32([[0.114], [0.587], [
        0.299]])
    out = np.matmul(img, M).clip(0, 255).astype(np.uint8)
    return out
