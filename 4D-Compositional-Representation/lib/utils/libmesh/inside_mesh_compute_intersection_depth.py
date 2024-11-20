def compute_intersection_depth(self, points, triangles):
    t1 = triangles[:, 0, :]
    t2 = triangles[:, 1, :]
    t3 = triangles[:, 2, :]
    v1 = t3 - t1
    v2 = t2 - t1
    normals = np.cross(v1, v2)
    alpha = np.sum(normals[:, :2] * (t1[:, :2] - points[:, :2]), axis=1)
    n_2 = normals[:, 2]
    t1_2 = t1[:, 2]
    s_n_2 = np.sign(n_2)
    abs_n_2 = np.abs(n_2)
    mask = abs_n_2 != 0
    depth_intersect = np.full(points.shape[0], np.nan)
    depth_intersect[mask] = t1_2[mask] * abs_n_2[mask] + alpha[mask] * s_n_2[
        mask]
    return depth_intersect, abs_n_2
