def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    R = Exp(r)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)
    c2w = convert3x4_4x4(c2w)
    return c2w
