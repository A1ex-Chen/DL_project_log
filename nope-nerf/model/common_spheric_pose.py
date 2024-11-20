def spheric_pose(theta, phi, radius):
    trans_t = lambda t: np.array([[1, 0, 0, 0], [0, 1, 0, 2 * mean_h], [0, 
        0, 1, -t]])
    rot_phi = lambda phi: np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi
        )], [0, np.sin(phi), np.cos(phi)]])
    rot_theta = lambda th: np.array([[np.cos(th), 0, -np.sin(th)], [0, 1, 0
        ], [np.sin(th), 0, np.cos(th)]])
    c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
    c2w = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]) @ c2w
    return c2w
