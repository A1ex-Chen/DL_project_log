def shape(x):
    """  projection """
    return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)
