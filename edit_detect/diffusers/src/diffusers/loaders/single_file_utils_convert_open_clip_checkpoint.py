def convert_open_clip_checkpoint(text_model, checkpoint, prefix=
    'cond_stage_model.model.'):
    text_model_dict = {}
    text_proj_key = prefix + 'text_projection'
    if text_proj_key in checkpoint:
        text_proj_dim = int(checkpoint[text_proj_key].shape[0])
    elif hasattr(text_model.config, 'projection_dim'):
        text_proj_dim = text_model.config.projection_dim
    else:
        text_proj_dim = LDM_OPEN_CLIP_TEXT_PROJECTION_DIM
    text_model_dict['text_model.embeddings.position_ids'
        ] = text_model.text_model.embeddings.get_buffer('position_ids')
    keys = list(checkpoint.keys())
    keys_to_ignore = SD_2_TEXT_ENCODER_KEYS_TO_IGNORE
    openclip_diffusers_ldm_map = DIFFUSERS_TO_LDM_MAPPING['openclip']['layers']
    for diffusers_key, ldm_key in openclip_diffusers_ldm_map.items():
        ldm_key = prefix + ldm_key
        if ldm_key not in checkpoint:
            continue
        if ldm_key in keys_to_ignore:
            continue
        if ldm_key.endswith('text_projection'):
            text_model_dict[diffusers_key] = checkpoint[ldm_key].T.contiguous()
        else:
            text_model_dict[diffusers_key] = checkpoint[ldm_key]
    for key in keys:
        if key in keys_to_ignore:
            continue
        if not key.startswith(prefix + 'transformer.'):
            continue
        diffusers_key = key.replace(prefix + 'transformer.', '')
        transformer_diffusers_to_ldm_map = DIFFUSERS_TO_LDM_MAPPING['openclip'
            ]['transformer']
        for new_key, old_key in transformer_diffusers_to_ldm_map.items():
            diffusers_key = diffusers_key.replace(old_key, new_key).replace(
                '.in_proj_weight', '').replace('.in_proj_bias', '')
        if key.endswith('.in_proj_weight'):
            weight_value = checkpoint.get(key)
            text_model_dict[diffusers_key + '.q_proj.weight'] = weight_value[
                :text_proj_dim, :].clone().detach()
            text_model_dict[diffusers_key + '.k_proj.weight'] = weight_value[
                text_proj_dim:text_proj_dim * 2, :].clone().detach()
            text_model_dict[diffusers_key + '.v_proj.weight'] = weight_value[
                text_proj_dim * 2:, :].clone().detach()
        elif key.endswith('.in_proj_bias'):
            weight_value = checkpoint.get(key)
            text_model_dict[diffusers_key + '.q_proj.bias'] = weight_value[:
                text_proj_dim].clone().detach()
            text_model_dict[diffusers_key + '.k_proj.bias'] = weight_value[
                text_proj_dim:text_proj_dim * 2].clone().detach()
            text_model_dict[diffusers_key + '.v_proj.bias'] = weight_value[
                text_proj_dim * 2:].clone().detach()
        else:
            text_model_dict[diffusers_key] = checkpoint.get(key)
    if not (hasattr(text_model, 'embeddings') and hasattr(text_model.
        embeddings.position_ids)):
        text_model_dict.pop('text_model.embeddings.position_ids', None)
    return text_model_dict
