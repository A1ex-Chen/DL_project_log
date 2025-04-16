def update_unet_attention_ldm_to_diffusers(ldm_keys, new_checkpoint,
    checkpoint, mapping):
    for ldm_key in ldm_keys:
        diffusers_key = ldm_key.replace(mapping['old'], mapping['new'])
        new_checkpoint[diffusers_key] = checkpoint.get(ldm_key)
