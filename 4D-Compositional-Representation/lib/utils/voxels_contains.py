def contains(self, points):
    nx = self.resolution
    points = (points - self.loc) / self.scale
    points_i = ((points + 0.5) * nx).astype(np.int32)
    i1, i2, i3 = points_i[..., 0], points_i[..., 1], points_i[..., 2]
    mask = (i1 >= 0) & (i2 >= 0) & (i3 >= 0) & (nx > i1) & (nx > i2) & (nx > i3
        )
    i1 = i1[mask]
    i2 = i2[mask]
    i3 = i3[mask]
    occ = np.zeros(points.shape[:-1], dtype=np.bool)
    occ[mask] = self.data[i1, i2, i3]
    return occ
