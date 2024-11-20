def blend_h(self, a, b, blend_extent):
    for x in range(min(a.shape[3], b.shape[3], blend_extent)):
        b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent
            ) + b[:, :, :, x] * (x / blend_extent)
    return b
