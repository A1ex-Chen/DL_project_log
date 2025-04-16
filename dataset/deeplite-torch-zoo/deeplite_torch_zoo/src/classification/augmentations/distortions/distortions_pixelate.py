@DISTORTION_REGISTRY.register('pixelate')
def pixelate(x, severity=1):
    w, h = x.shape[0], x.shape[1]
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    x = cv2.resize(x, (int(w * c), int(h * c)))
    x = cv2.resize(x, (w, h))
    return x
