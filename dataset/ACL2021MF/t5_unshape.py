def unshape(x):
    """  compute context """
    return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)
