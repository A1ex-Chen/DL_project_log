def convert_open_clap_checkpoint(checkpoint):
    """
    Takes a state dict and returns a converted CLAP checkpoint.
    """
    model_state_dict = {}
    model_key = 'cond_stage_model.model.text_'
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(model_key):
            model_state_dict[key.replace(model_key, 'text_')] = checkpoint.get(
                key)
    new_checkpoint = {}
    sequential_layers_pattern = '.*sequential.(\\d+).*'
    text_projection_pattern = '.*_projection.(\\d+).*'
    for key, value in model_state_dict.items():
        if key.split('.')[0] in CLAP_KEYS_TO_IGNORE:
            continue
        for key_to_modify, new_key in CLAP_KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        if re.match(sequential_layers_pattern, key):
            sequential_layer = re.match(sequential_layers_pattern, key).group(1
                )
            key = key.replace(f'sequential.{sequential_layer}.',
                f'layers.{int(sequential_layer) // 3}.linear.')
        elif re.match(text_projection_pattern, key):
            projecton_layer = int(re.match(text_projection_pattern, key).
                group(1))
            transformers_projection_layer = 1 if projecton_layer == 0 else 2
            key = key.replace(f'_projection.{projecton_layer}.',
                f'_projection.linear{transformers_projection_layer}.')
        if 'audio' and 'qkv' in key:
            mixed_qkv = value
            qkv_dim = mixed_qkv.size(0) // 3
            query_layer = mixed_qkv[:qkv_dim]
            key_layer = mixed_qkv[qkv_dim:qkv_dim * 2]
            value_layer = mixed_qkv[qkv_dim * 2:]
            new_checkpoint[key.replace('qkv', 'query')] = query_layer
            new_checkpoint[key.replace('qkv', 'key')] = key_layer
            new_checkpoint[key.replace('qkv', 'value')] = value_layer
        else:
            new_checkpoint[key] = value
    return new_checkpoint
