def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + torch.sin(norm_r) / norm_r * skew_r + (1 - torch.cos(norm_r)
        ) / norm_r ** 2 * (skew_r @ skew_r)
    return R
