def create_spheric_poses(radius, mean_h, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
        mean_h: mean camera height
    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([[1, 0, 0, 0], [0, 1, 0, 2 * mean_h],
            [0, 0, 1, -t]])
        rot_phi = lambda phi: np.array([[1, 0, 0], [0, np.cos(phi), -np.sin
            (phi)], [0, np.sin(phi), np.cos(phi)]])
        rot_theta = lambda th: np.array([[np.cos(th), 0, -np.sin(th)], [0, 
            1, 0], [np.sin(th), 0, np.cos(th)]])
        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]) @ c2w
        return c2w
    spheric_poses = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi / 12, radius)]
    return np.stack(spheric_poses, 0)
