def enclosing_box3d_convex_hull(corners1, corners2, nums_k2, mask,
    enclosing_vols=None):
    B, K1 = corners1.shape[0], corners1.shape[1]
    _, K2 = corners2.shape[0], corners2.shape[1]
    if enclosing_vols is None:
        enclosing_vols = np.zeros((B, K1, K2)).astype(np.float32)
    for b in range(B):
        for k1 in range(K1):
            for k2 in range(K2):
                if nums_k2 is not None and k2 >= nums_k2[b]:
                    break
                if mask is not None and mask[b, k1, k2] <= 0:
                    continue
                hull = ConvexHull(np.vstack([corners1[b, k1], corners2[b, k2]])
                    )
                enclosing_vols[b, k1, k2] = hull.volume
    return enclosing_vols
