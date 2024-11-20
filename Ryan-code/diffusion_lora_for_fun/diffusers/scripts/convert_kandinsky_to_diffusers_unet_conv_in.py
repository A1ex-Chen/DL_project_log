def unet_conv_in(checkpoint):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update({'conv_in.weight': checkpoint[
        'input_blocks.0.0.weight'], 'conv_in.bias': checkpoint[
        'input_blocks.0.0.bias']})
    return diffusers_checkpoint
