def transpose_for_scores(self, x):
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)