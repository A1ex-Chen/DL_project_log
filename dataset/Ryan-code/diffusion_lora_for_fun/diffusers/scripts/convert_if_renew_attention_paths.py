def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item
        if 'qkv' in new_item:
            continue
        if 'encoder_kv' in new_item:
            continue
        new_item = new_item.replace('norm.weight', 'group_norm.weight')
        new_item = new_item.replace('norm.bias', 'group_norm.bias')
        new_item = new_item.replace('proj_out.weight', 'to_out.0.weight')
        new_item = new_item.replace('proj_out.bias', 'to_out.0.bias')
        new_item = new_item.replace('norm_encoder.weight', 'norm_cross.weight')
        new_item = new_item.replace('norm_encoder.bias', 'norm_cross.bias')
        new_item = shave_segments(new_item, n_shave_prefix_segments=
            n_shave_prefix_segments)
        mapping.append({'old': old_item, 'new': new_item})
    return mapping
