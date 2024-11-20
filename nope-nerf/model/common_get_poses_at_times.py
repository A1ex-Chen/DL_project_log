def get_poses_at_times(c2ws, input_times, target_times):
    trans = c2ws[:, :3, 3:]
    rots = c2ws[:, :3, :3]
    N_target = len(target_times)
    rots = R.from_matrix(rots)
    slerp = Slerp(input_times, rots)
    target_rots = torch.tensor(slerp(target_times).as_matrix().astype(np.
        float32))
    target_trans = interp_t(trans, input_times, target_times)
    target_poses = torch.cat([target_rots, target_trans], dim=2)
    target_poses = convert3x4_4x4(target_poses)
    return target_poses
