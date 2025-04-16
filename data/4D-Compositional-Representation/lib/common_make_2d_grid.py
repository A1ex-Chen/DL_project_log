def make_2d_grid(bb_min, bb_max, shape):
    size = shape[0] * shape[1]
    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pxs = pxs.view(-1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys], dim=1)
    return p
