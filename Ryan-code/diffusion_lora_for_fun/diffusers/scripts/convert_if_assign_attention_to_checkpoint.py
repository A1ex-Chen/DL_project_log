def assign_attention_to_checkpoint(new_checkpoint, unet_state_dict,
    old_path, new_path, config):
    qkv_weight = unet_state_dict.pop(f'{old_path}.qkv.weight')
    qkv_weight = qkv_weight[:, :, 0]
    qkv_bias = unet_state_dict.pop(f'{old_path}.qkv.bias')
    is_cross_attn_only = 'only_cross_attention' in config and config[
        'only_cross_attention']
    split = 1 if is_cross_attn_only else 3
    weights, bias = split_attentions(weight=qkv_weight, bias=qkv_bias,
        split=split, chunk_size=config['attention_head_dim'])
    if is_cross_attn_only:
        query_weight, q_bias = weights, bias
        new_checkpoint[f'{new_path}.to_q.weight'] = query_weight[0]
        new_checkpoint[f'{new_path}.to_q.bias'] = q_bias[0]
    else:
        [query_weight, key_weight, value_weight], [q_bias, k_bias, v_bias
            ] = weights, bias
        new_checkpoint[f'{new_path}.to_q.weight'] = query_weight
        new_checkpoint[f'{new_path}.to_q.bias'] = q_bias
        new_checkpoint[f'{new_path}.to_k.weight'] = key_weight
        new_checkpoint[f'{new_path}.to_k.bias'] = k_bias
        new_checkpoint[f'{new_path}.to_v.weight'] = value_weight
        new_checkpoint[f'{new_path}.to_v.bias'] = v_bias
    encoder_kv_weight = unet_state_dict.pop(f'{old_path}.encoder_kv.weight')
    encoder_kv_weight = encoder_kv_weight[:, :, 0]
    encoder_kv_bias = unet_state_dict.pop(f'{old_path}.encoder_kv.bias')
    [encoder_k_weight, encoder_v_weight], [encoder_k_bias, encoder_v_bias
        ] = split_attentions(weight=encoder_kv_weight, bias=encoder_kv_bias,
        split=2, chunk_size=config['attention_head_dim'])
    new_checkpoint[f'{new_path}.add_k_proj.weight'] = encoder_k_weight
    new_checkpoint[f'{new_path}.add_k_proj.bias'] = encoder_k_bias
    new_checkpoint[f'{new_path}.add_v_proj.weight'] = encoder_v_weight
    new_checkpoint[f'{new_path}.add_v_proj.bias'] = encoder_v_bias
