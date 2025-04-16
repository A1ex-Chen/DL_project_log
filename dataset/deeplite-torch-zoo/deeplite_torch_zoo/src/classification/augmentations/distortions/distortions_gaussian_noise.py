@DISTORTION_REGISTRY.register('gaussian_noise')
def gaussian_noise(x, severity=1):
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255
