def convert_open_clip_checkpoint(checkpoint, config_name, prefix=
    'cond_stage_model.model.', has_projection=False, local_files_only=False,
    **config_kwargs):
    try:
        config = CLIPTextConfig.from_pretrained(config_name, **
            config_kwargs, local_files_only=local_files_only)
    except Exception:
        raise ValueError(
            f"With local_files_only set to {local_files_only}, you must first locally save the configuration in the following path: '{config_name}'."
            )
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        text_model = CLIPTextModelWithProjection(config
            ) if has_projection else CLIPTextModel(config)
    keys = list(checkpoint.keys())
    keys_to_ignore = []
    if (config_name == 'stabilityai/stable-diffusion-2' and config.
        num_hidden_layers == 23):
        keys_to_ignore += [k for k in keys if k.startswith(
            'cond_stage_model.model.transformer.resblocks.23')]
        keys_to_ignore += ['cond_stage_model.model.text_projection']
    text_model_dict = {}
    if prefix + 'text_projection' in checkpoint:
        d_model = int(checkpoint[prefix + 'text_projection'].shape[0])
    else:
        d_model = 1024
    text_model_dict['text_model.embeddings.position_ids'
        ] = text_model.text_model.embeddings.get_buffer('position_ids')
    for key in keys:
        if key in keys_to_ignore:
            continue
        if key[len(prefix):] in textenc_conversion_map:
            if key.endswith('text_projection'):
                value = checkpoint[key].T.contiguous()
            else:
                value = checkpoint[key]
            text_model_dict[textenc_conversion_map[key[len(prefix):]]] = value
        if key.startswith(prefix + 'transformer.'):
            new_key = key[len(prefix + 'transformer.'):]
            if new_key.endswith('.in_proj_weight'):
                new_key = new_key[:-len('.in_proj_weight')]
                new_key = textenc_pattern.sub(lambda m: protected[re.escape
                    (m.group(0))], new_key)
                text_model_dict[new_key + '.q_proj.weight'] = checkpoint[key][
                    :d_model, :]
                text_model_dict[new_key + '.k_proj.weight'] = checkpoint[key][
                    d_model:d_model * 2, :]
                text_model_dict[new_key + '.v_proj.weight'] = checkpoint[key][
                    d_model * 2:, :]
            elif new_key.endswith('.in_proj_bias'):
                new_key = new_key[:-len('.in_proj_bias')]
                new_key = textenc_pattern.sub(lambda m: protected[re.escape
                    (m.group(0))], new_key)
                text_model_dict[new_key + '.q_proj.bias'] = checkpoint[key][:
                    d_model]
                text_model_dict[new_key + '.k_proj.bias'] = checkpoint[key][
                    d_model:d_model * 2]
                text_model_dict[new_key + '.v_proj.bias'] = checkpoint[key][
                    d_model * 2:]
            else:
                new_key = textenc_pattern.sub(lambda m: protected[re.escape
                    (m.group(0))], new_key)
                text_model_dict[new_key] = checkpoint[key]
    if is_accelerate_available():
        for param_name, param in text_model_dict.items():
            set_module_tensor_to_device(text_model, param_name, 'cpu',
                value=param)
    else:
        if not (hasattr(text_model, 'embeddings') and hasattr(text_model.
            embeddings.position_ids)):
            text_model_dict.pop('text_model.embeddings.position_ids', None)
        text_model.load_state_dict(text_model_dict)
    return text_model
