def unet_conv_out(checkpoint):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update({'conv_out.weight': checkpoint[
        'out.2.weight'], 'conv_out.bias': checkpoint['out.2.bias']})
    return diffusers_checkpoint
