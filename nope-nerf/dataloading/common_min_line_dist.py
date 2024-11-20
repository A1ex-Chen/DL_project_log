def min_line_dist(rays_o, rays_d):
    A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
    b_i = -A_i @ rays_o
    pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @
        A_i).mean(0)) @ b_i.mean(0))
    return pt_mindist
