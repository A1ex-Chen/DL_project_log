def alignSE3Single(p_es, p_gt, q_es, q_gt):
    """
    Calculate SE3 transformation R and t so that:
        gt = R * est + t
    Using only the first poses of est and gt
    """
    p_es_0, q_es_0 = p_es[0, :], q_es[0, :]
    p_gt_0, q_gt_0 = p_gt[0, :], q_gt[0, :]
    g_rot = tfs.quaternion_matrix(q_gt_0)
    g_rot = g_rot[0:3, 0:3]
    est_rot = tfs.quaternion_matrix(q_es_0)
    est_rot = est_rot[0:3, 0:3]
    R = np.dot(g_rot, np.transpose(est_rot))
    t = p_gt_0 - np.dot(R, p_es_0)
    return R, t
