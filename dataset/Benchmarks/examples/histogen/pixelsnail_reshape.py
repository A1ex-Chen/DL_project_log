def reshape(input):
    return input.view(batch, -1, self.n_head, self.dim_head).transpose(1, 2)
