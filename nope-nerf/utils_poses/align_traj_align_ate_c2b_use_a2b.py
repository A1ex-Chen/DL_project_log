def align_ate_c2b_use_a2b(traj_a, traj_b, traj_c=None):
    """Align c to b using the sim3 from a to b.
    :param traj_a:  (N0, 3/4, 4) torch tensor
    :param traj_b:  (N0, 3/4, 4) torch tensor
    :param traj_c:  None or (N1, 3/4, 4) torch tensor
    :return:        (N1, 4,   4) torch tensor
    """
    device = traj_a.device
    if traj_c is None:
        traj_c = traj_a.clone()
    traj_a = traj_a.float().cpu().numpy()
    traj_b = traj_b.float().cpu().numpy()
    traj_c = traj_c.float().cpu().numpy()
    R_a = traj_a[:, :3, :3]
    t_a = traj_a[:, :3, 3]
    quat_a = SO3_to_quat(R_a)
    R_b = traj_b[:, :3, :3]
    t_b = traj_b[:, :3, 3]
    quat_b = SO3_to_quat(R_b)
    s, R, t = alignTrajectory(t_a, t_b, quat_a, quat_b, method='sim3')
    R = R[None, :, :].astype(np.float32)
    t = t[None, :, None].astype(np.float32)
    s = float(s)
    R_c = traj_c[:, :3, :3]
    t_c = traj_c[:, :3, 3:4]
    R_c_aligned = R @ R_c
    t_c_aligned = s * (R @ t_c) + t
    traj_c_aligned = np.concatenate([R_c_aligned, t_c_aligned], axis=2)
    traj_c_aligned = convert3x4_4x4(traj_c_aligned)
    traj_c_aligned = torch.from_numpy(traj_c_aligned).to(device)
    return traj_c_aligned
