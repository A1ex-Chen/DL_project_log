def convert_ldm_clip_checkpoint(checkpoint):
    keys = list(checkpoint.keys())
    text_model_dict = {}
    remove_prefixes = LDM_CLIP_PREFIX_TO_REMOVE
    for key in keys:
        for prefix in remove_prefixes:
            if key.startswith(prefix):
                diffusers_key = key.replace(prefix, '')
                text_model_dict[diffusers_key] = checkpoint.get(key)
    return text_model_dict
