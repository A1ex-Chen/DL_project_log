def _split_hidden_size_dim(self, x, num_attn_heads, attn_head_size):
    """
        splits hidden_size dim into attn_head_size and num_attn_heads
        """
    new_x_shape = x.size()[:-1] + (num_attn_heads, attn_head_size)
    x = x.view(*new_x_shape)
    return x.transpose(2, 1)
