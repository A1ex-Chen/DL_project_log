def _reshape(self, tensor, first_dim, batch_size):
    return tensor.reshape(first_dim, batch_size * self.num_attn_heads, self
        .head_dim).transpose(0, 1)
