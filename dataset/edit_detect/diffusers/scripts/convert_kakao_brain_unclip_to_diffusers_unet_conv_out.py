def unet_conv_out(checkpoint, original_unet_prefix):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update({'conv_out.weight': checkpoint[
        f'{original_unet_prefix}.out.2.weight'], 'conv_out.bias':
        checkpoint[f'{original_unet_prefix}.out.2.bias']})
    return diffusers_checkpoint
