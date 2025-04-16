def update_unet_resnet_ldm_to_diffusers(ldm_keys, new_checkpoint,
    checkpoint, mapping=None):
    for ldm_key in ldm_keys:
        diffusers_key = ldm_key.replace('in_layers.0', 'norm1').replace(
            'in_layers.2', 'conv1').replace('out_layers.0', 'norm2').replace(
            'out_layers.3', 'conv2').replace('emb_layers.1', 'time_emb_proj'
            ).replace('skip_connection', 'conv_shortcut')
        if mapping:
            diffusers_key = diffusers_key.replace(mapping['old'], mapping[
                'new'])
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)
