def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]
    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([0.2 * np.cos(theta), -0.2 * np.
            sin(theta), -np.sin(theta * zrate) * 0.1, 1.0]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
