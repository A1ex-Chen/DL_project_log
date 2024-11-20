def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item
        new_item = new_item.replace('norm.weight', 'group_norm.weight')
        new_item = new_item.replace('norm.bias', 'group_norm.bias')
        new_item = new_item.replace('q.weight', 'to_q.weight')
        new_item = new_item.replace('q.bias', 'to_q.bias')
        new_item = new_item.replace('k.weight', 'to_k.weight')
        new_item = new_item.replace('k.bias', 'to_k.bias')
        new_item = new_item.replace('v.weight', 'to_v.weight')
        new_item = new_item.replace('v.bias', 'to_v.bias')
        new_item = new_item.replace('proj_out.weight', 'to_out.0.weight')
        new_item = new_item.replace('proj_out.bias', 'to_out.0.bias')
        new_item = shave_segments(new_item, n_shave_prefix_segments=
            n_shave_prefix_segments)
        mapping.append({'old': old_item, 'new': new_item})
    return mapping
