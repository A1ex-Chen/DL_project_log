def create_custom_forward(module):

    def custom_forward(*inputs):
        return tuple(output for output in module(*inputs, use_cache,
            output_attentions))
    return custom_forward
