def custom_forward(*inputs):
    return tuple(module(*inputs, use_cache, output_attentions))
