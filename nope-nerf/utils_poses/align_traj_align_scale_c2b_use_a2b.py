def align_scale_c2b_use_a2b(traj_a, traj_b, traj_c=None):
    """Scale c to b using the scale from a to b.
    :param traj_a:      (N0, 3/4, 4) torch tensor
    :param traj_b:      (N0, 3/4, 4) torch tensor
    :param traj_c:      None or (N1, 3/4, 4) torch tensor
    :return:
        scaled_traj_c   (N1, 4, 4)   torch tensor
        scale           scalar
    """
    if traj_c is None:
        traj_c = traj_a.clone()
    t_a = traj_a[:, :3, 3]
    t_b = traj_b[:, :3, 3]
    scale_a2b = pts_dist_max(t_b) / pts_dist_max(t_a)
    traj_c[:, :3, 3] *= scale_a2b
    if traj_c.shape[1] == 3:
        traj_c = convert3x4_4x4(traj_c)
    return traj_c, scale_a2b
