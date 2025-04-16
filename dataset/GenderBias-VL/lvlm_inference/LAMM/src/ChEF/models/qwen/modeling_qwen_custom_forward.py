def custom_forward(*inputs):
    return module(*inputs, use_cache, output_attentions)
