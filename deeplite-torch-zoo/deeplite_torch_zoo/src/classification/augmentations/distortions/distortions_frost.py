@DISTORTION_REGISTRY.register('frost')
def frost(x, severity=1):
    HERE = pathlib.Path(__file__).parent
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][
        severity - 1]
    w, h = x.shape[0], x.shape[1]
    idx = np.random.randint(5)
    filename = [HERE / './frost_imgs/frost1.png', HERE /
        './frost_imgs/frost2.png', HERE / './frost_imgs/frost3.png', HERE /
        './frost_imgs/frost4.jpg', HERE / './frost_imgs/frost5.jpg', HERE /
        './frost_imgs/frost6.jpg'][idx]
    frost = cv2.imread(str(filename))
    frost = cv2.resize(frost, (h, w))
    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)
