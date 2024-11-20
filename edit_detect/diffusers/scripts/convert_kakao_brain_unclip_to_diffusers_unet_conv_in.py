def unet_conv_in(checkpoint, original_unet_prefix):
    diffusers_checkpoint = {}
    diffusers_checkpoint.update({'conv_in.weight': checkpoint[
        f'{original_unet_prefix}.input_blocks.0.0.weight'], 'conv_in.bias':
        checkpoint[f'{original_unet_prefix}.input_blocks.0.0.bias']})
    return diffusers_checkpoint
