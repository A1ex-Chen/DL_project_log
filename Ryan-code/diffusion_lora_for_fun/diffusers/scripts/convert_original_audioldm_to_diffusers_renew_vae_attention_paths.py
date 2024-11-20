def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item
        new_item = new_item.replace('norm.weight', 'group_norm.weight')
        new_item = new_item.replace('norm.bias', 'group_norm.bias')
        new_item = new_item.replace('q.weight', 'query.weight')
        new_item = new_item.replace('q.bias', 'query.bias')
        new_item = new_item.replace('k.weight', 'key.weight')
        new_item = new_item.replace('k.bias', 'key.bias')
        new_item = new_item.replace('v.weight', 'value.weight')
        new_item = new_item.replace('v.bias', 'value.bias')
        new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
        new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')
        new_item = shave_segments(new_item, n_shave_prefix_segments=
            n_shave_prefix_segments)
        mapping.append({'old': old_item, 'new': new_item})
    return mapping
