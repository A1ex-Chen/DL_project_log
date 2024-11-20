def alignPositionYawSingle(p_es, p_gt, q_es, q_gt):
    """
    calcualte the 4DOF transformation: yaw R and translation t so that:
        gt = R * est + t
    """
    p_es_0, q_es_0 = p_es[0, :], q_es[0, :]
    p_gt_0, q_gt_0 = p_gt[0, :], q_gt[0, :]
    g_rot = tfs.quaternion_matrix(q_gt_0)
    g_rot = g_rot[0:3, 0:3]
    est_rot = tfs.quaternion_matrix(q_es_0)
    est_rot = est_rot[0:3, 0:3]
    C_R = np.dot(est_rot, g_rot.transpose())
    theta = align.get_best_yaw(C_R)
    R = align.rot_z(theta)
    t = p_gt_0 - np.dot(R, p_es_0)
    return R, t
