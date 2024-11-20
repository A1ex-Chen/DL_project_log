def custom_forward(*inputs):
    return module(*inputs, past_key_value, output_attentions)
