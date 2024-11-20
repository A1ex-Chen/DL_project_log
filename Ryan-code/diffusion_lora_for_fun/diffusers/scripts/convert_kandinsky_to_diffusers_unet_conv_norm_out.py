def unet_conv_norm_out(checkpoint):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update({'conv_norm_out.weight': checkpoint[
        'out.0.weight'], 'conv_norm_out.bias': checkpoint['out.0.bias']})
    return diffusers_checkpoint
