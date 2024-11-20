def create_custom_forward(module):

    def custom_forward(*inputs):
        return module(*inputs, output_attentions, None, im_mask)
    return custom_forward
