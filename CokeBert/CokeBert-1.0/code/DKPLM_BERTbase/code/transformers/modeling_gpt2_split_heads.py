def split_heads(self, x, k=False):
    new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
    x = x.view(*new_x_shape)
    if k:
        return x.permute(0, 2, 3, 1)
    else:
        return x.permute(0, 2, 1, 3)
