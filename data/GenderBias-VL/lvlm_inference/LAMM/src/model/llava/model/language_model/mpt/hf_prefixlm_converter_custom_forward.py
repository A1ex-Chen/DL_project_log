def custom_forward(*inputs):
    return module(*inputs, use_cache=use_cache, output_attentions=
        output_attentions)
