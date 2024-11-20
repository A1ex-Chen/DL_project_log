def blend_v(self, a, b, blend_extent):
    for y in range(min(a.shape[2], b.shape[2], blend_extent)):
        b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent
            ) + b[:, :, y, :] * (y / blend_extent)
    return b
