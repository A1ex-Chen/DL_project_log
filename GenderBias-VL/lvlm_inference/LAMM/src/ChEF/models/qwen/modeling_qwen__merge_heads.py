def _merge_heads(self, tensor, num_heads, attn_head_size):
    tensor = tensor.contiguous()
    new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
    return tensor.view(new_shape)
