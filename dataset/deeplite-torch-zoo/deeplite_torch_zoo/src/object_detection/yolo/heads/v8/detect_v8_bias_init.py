def bias_init(self):
    m = self
    for a, b, s in zip(m.cv2, m.cv3, m.stride):
        a[-1].bias.data[:] = 1.0
        b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
