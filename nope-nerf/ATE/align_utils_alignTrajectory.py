def alignTrajectory(p_es, p_gt, q_es, q_gt, method, n_aligned=-1):
    """
    calculate s, R, t so that:
        gt = R * s * est + t
    method can be: sim3, se3, posyaw, none;
    n_aligned: -1 means using all the frames
    """
    assert p_es.shape[1] == 3
    assert p_gt.shape[1] == 3
    assert q_es.shape[1] == 4
    assert q_gt.shape[1] == 4
    s = 1
    R = None
    t = None
    if method == 'sim3':
        assert n_aligned >= 2 or n_aligned == -1, 'sim3 uses at least 2 frames'
        s, R, t = alignSIM3(p_es, p_gt, q_es, q_gt, n_aligned)
    elif method == 'se3':
        R, t = alignSE3(p_es, p_gt, q_es, q_gt, n_aligned)
    elif method == 'posyaw':
        R, t = alignPositionYaw(p_es, p_gt, q_es, q_gt, n_aligned)
    elif method == 'none':
        R = np.identity(3)
        t = np.zeros((3,))
    else:
        assert False, 'unknown alignment method'
    return s, R, t
