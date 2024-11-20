@DISTORTION_REGISTRY.register('contrast')
def contrast(x, severity=1):
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    x = np.array(x) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255
