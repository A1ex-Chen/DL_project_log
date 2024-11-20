def custom_forward(*inputs):
    return module(*inputs, output_attentions, None, im_mask)
