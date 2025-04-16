@DISTORTION_REGISTRY.register('gaussian_blur')
def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]
    x = gaussian(np.array(x) / 255.0, sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255
