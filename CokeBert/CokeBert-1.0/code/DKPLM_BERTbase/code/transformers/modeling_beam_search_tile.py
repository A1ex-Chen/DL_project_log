def tile(x, count, dim=0):
    """
    Tiles `x` along dimension `dim` `count` times.

    Example:
        >> ex = torch.tensor([1,2],[3,4])
        >> tile(ex, 2, 0)
        torch.Tensor([[1,2],[1,2],[3,4],[3,4]])
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1).transpose(0, 1).repeat(count, 1).transpose(0, 1
        ).contiguous().view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x
