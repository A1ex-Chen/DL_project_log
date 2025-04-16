def _get_colormap(n):

    def bitget(byteval, idx):
        return byteval & 1 << idx != 0
    cmap = np.zeros((n, 3), dtype='uint8')
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | bitget(c, 0) << 7 - j
            g = g | bitget(c, 1) << 7 - j
            b = b | bitget(c, 2) << 7 - j
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    return cmap
