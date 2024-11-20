def spherify_poses(poses, bds):
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-
        1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)
    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]
            ) @ A_i).mean(0)) @ b_i.mean(0))
        return pt_mindist
    pt_mindist = min_line_dist(rays_o, rays_d)
    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)
    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)
    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,
        :3, :4])
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc
    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []
    for th in np.linspace(0.0, 2.0 * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th
            ), zh])
        up = np.array([0, 0, -1.0])
        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)
        new_poses.append(p)
    new_poses = np.stack(new_poses, 0)
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:
        ], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:, :3, :4], np.broadcast_to(
        poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)
    return poses_reset, new_poses, bds
