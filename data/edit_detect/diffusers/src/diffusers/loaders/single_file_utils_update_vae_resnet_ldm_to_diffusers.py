def update_vae_resnet_ldm_to_diffusers(keys, new_checkpoint, checkpoint,
    mapping):
    for ldm_key in keys:
        diffusers_key = ldm_key.replace(mapping['old'], mapping['new']
            ).replace('nin_shortcut', 'conv_shortcut')
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)
