def unet_conv_norm_out(checkpoint, original_unet_prefix):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update({'conv_norm_out.weight': checkpoint[
        f'{original_unet_prefix}.out.0.weight'], 'conv_norm_out.bias':
        checkpoint[f'{original_unet_prefix}.out.0.bias']})
    return diffusers_checkpoint
