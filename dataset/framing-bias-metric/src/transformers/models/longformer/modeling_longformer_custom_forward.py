def custom_forward(*inputs):
    return module(*inputs, is_global_attn)
