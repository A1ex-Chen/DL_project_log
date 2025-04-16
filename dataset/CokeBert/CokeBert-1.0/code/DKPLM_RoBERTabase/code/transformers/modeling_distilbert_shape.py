def shape(x):
    """ separate heads """
    return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
