def create_custom_forward(module):

    def custom_forward(*inputs):
        return module(*inputs, use_cache=use_cache, output_attentions=
            output_attentions)
    return custom_forward
