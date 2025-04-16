@DISTORTION_REGISTRY.register('fog')
def fog(x, severity=1):
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]
    w, h = x.shape[0], x.shape[1]
    x = np.array(x) / 255.0
    max_val = x.max()
    x += c[0] * plasma_fractal(mapsize=max(w, h), wibbledecay=c[1])[:w, :h][
        ..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
