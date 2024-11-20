def fix_Rt_camera(Rt, loc, scale):
    """ Fixes Rt camera matrix.

    Args:
        Rt (tensor): Rt camera matrix
        loc (tensor): location
        scale (float): scale
    """
    batch_size = Rt.size(0)
    R = Rt[:, :, :3]
    t = Rt[:, :, 3:]
    scale = scale.view(batch_size, 1, 1)
    R_new = R * scale
    t_new = t + R @ loc.unsqueeze(2)
    Rt_new = torch.cat([R_new, t_new], dim=2)
    assert Rt_new.size() == (batch_size, 3, 4)
    return Rt_new
