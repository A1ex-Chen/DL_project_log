def shape(x):
    x = x.view(bs, -1, self.num_heads, self.dim_per_head)
    x = x.transpose(1, 2)
    x = x.reshape(bs * self.num_heads, -1, self.dim_per_head)
    x = x.transpose(1, 2)
    return x
