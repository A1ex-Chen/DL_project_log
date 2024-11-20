def generate_spiral_nerf(learned_poses, bds, N_novel_views, hwf):
    learned_poses_ = np.concatenate((learned_poses[:, :3, :4].detach().cpu(
        ).numpy(), hwf[:len(learned_poses)]), axis=-1)
    c2w = poses_avg(learned_poses_)
    print('recentered', c2w.shape)
    up = normalize(learned_poses_[:, :3, 1].sum(0))
    close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
    dt = 0.75
    mean_dz = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)
    focal = mean_dz
    shrink_factor = 0.8
    zdelta = close_depth * 0.2
    tt = learned_poses_[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_rots = 2
    c2ws = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=0.5,
        rots=N_rots, N=N_novel_views)
    c2ws = torch.tensor(np.stack(c2ws).astype(np.float32))
    c2ws = c2ws[:, :3, :4]
    return c2ws
