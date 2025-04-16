def alignPositionYaw(p_es, p_gt, q_es, q_gt, n_aligned=1):
    if n_aligned == 1:
        R, t = alignPositionYawSingle(p_es, p_gt, q_es, q_gt)
        return R, t
    else:
        idxs = _getIndices(n_aligned, p_es.shape[0])
        est_pos = p_es[idxs, 0:3]
        gt_pos = p_gt[idxs, 0:3]
        _, R, t = align.align_umeyama(gt_pos, est_pos, known_scale=True,
            yaw_only=True)
        t = np.array(t)
        t = t.reshape((3,))
        R = np.array(R)
        return R, t
