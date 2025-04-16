def pts_dist_max(pts):
    """
    :param pts:  (N, 3) torch or np
    :return:     scalar
    """
    if torch.is_tensor(pts):
        dist = pts.unsqueeze(0) - pts.unsqueeze(1)
        dist = dist[0]
        dist = dist.norm(dim=1)
        max_dist = dist.max()
    else:
        dist = pts[None, :, :] - pts[:, None, :]
        dist = dist[0]
        dist = np.linalg.norm(dist, axis=1)
        max_dist = dist.max()
    return max_dist
