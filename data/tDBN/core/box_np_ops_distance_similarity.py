@numba.jit(nopython=True)
def distance_similarity(points, qpoints, dist_norm, with_rotation=False,
    rot_alpha=0.5):
    N = points.shape[0]
    K = qpoints.shape[0]
    dists = np.zeros((N, K), dtype=points.dtype)
    rot_alpha_1 = 1 - rot_alpha
    for k in range(K):
        for n in range(N):
            if np.abs(points[n, 0] - qpoints[k, 0]) <= dist_norm:
                if np.abs(points[n, 1] - qpoints[k, 1]) <= dist_norm:
                    dist = np.sum((points[n, :2] - qpoints[k, :2]) ** 2)
                    dist_normed = min(dist / dist_norm, dist_norm)
                    if with_rotation:
                        dist_rot = np.abs(np.sin(points[n, -1] - qpoints[k,
                            -1]))
                        dists[n, k] = (1 - rot_alpha_1 * dist_normed - 
                            rot_alpha * dist_rot)
                    else:
                        dists[n, k] = 1 - dist_normed
    return dists
