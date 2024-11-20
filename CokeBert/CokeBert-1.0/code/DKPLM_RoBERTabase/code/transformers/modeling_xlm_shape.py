def shape(x):
    """  projection """
    return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
