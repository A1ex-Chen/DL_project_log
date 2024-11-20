def alignSIM3(p_es, p_gt, q_es, q_gt, n_aligned=-1):
    """
    calculate s, R, t so that:
        gt = R * s * est + t
    """
    idxs = _getIndices(n_aligned, p_es.shape[0])
    est_pos = p_es[idxs, 0:3]
    gt_pos = p_gt[idxs, 0:3]
    s, R, t = align.align_umeyama(gt_pos, est_pos)
    return s, R, t
