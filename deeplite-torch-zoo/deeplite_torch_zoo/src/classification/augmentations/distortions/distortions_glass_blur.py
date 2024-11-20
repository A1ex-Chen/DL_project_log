@DISTORTION_REGISTRY.register('glass_blur')
def glass_blur(x, severity=1):
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][
        severity - 1]
    x = np.uint8(gaussian(np.array(x) / 255.0, sigma=c[0], channel_axis=-1) *
        255)
    h_image, w_image = x.shape[0], x.shape[1]
    for i in range(c[2]):
        for h in range(h_image - c[1], c[1], -1):
            for w in range(w_image - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
    return np.clip(gaussian(x / 255.0, sigma=c[0], channel_axis=-1), 0, 1
        ) * 255
