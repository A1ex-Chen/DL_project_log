def check_triangles(self, points, triangles):
    contains = np.zeros(points.shape[0], dtype=np.bool)
    A = triangles[:, :2] - triangles[:, 2:]
    A = A.transpose([0, 2, 1])
    y = points - triangles[:, 2]
    detA = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    mask = np.abs(detA) != 0.0
    A = A[mask]
    y = y[mask]
    detA = detA[mask]
    s_detA = np.sign(detA)
    abs_detA = np.abs(detA)
    u = (A[:, 1, 1] * y[:, 0] - A[:, 0, 1] * y[:, 1]) * s_detA
    v = (-A[:, 1, 0] * y[:, 0] + A[:, 0, 0] * y[:, 1]) * s_detA
    sum_uv = u + v
    contains[mask] = (0 < u) & (u < abs_detA) & (0 < v) & (v < abs_detA) & (
        0 < sum_uv) & (sum_uv < abs_detA)
    return contains
