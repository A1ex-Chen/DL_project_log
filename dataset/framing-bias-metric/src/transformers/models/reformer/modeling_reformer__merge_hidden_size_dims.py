def _merge_hidden_size_dims(self, x, num_attn_heads, attn_head_size):
    """
        merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
    x = x.permute(0, 2, 1, 3)
    return torch.reshape(x, (x.size()[0], -1, num_attn_heads * attn_head_size))
