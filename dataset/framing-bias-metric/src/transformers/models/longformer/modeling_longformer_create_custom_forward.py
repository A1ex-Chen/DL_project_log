def create_custom_forward(module):

    def custom_forward(*inputs):
        return module(*inputs, is_global_attn)
    return custom_forward
