@DISTORTION_REGISTRY.register('impulse_noise')
def impulse_noise(x, severity=1):
    c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255.0, mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255
