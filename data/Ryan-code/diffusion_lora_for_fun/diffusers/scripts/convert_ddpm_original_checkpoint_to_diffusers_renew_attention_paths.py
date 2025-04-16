def renew_attention_paths(old_list, n_shave_prefix_segments=0, in_mid=False):
    mapping = []
    for old_item in old_list:
        new_item = old_item
        if not in_mid:
            new_item = new_item.replace('attn', 'attentions')
        new_item = new_item.replace('.k.', '.key.')
        new_item = new_item.replace('.v.', '.value.')
        new_item = new_item.replace('.q.', '.query.')
        new_item = new_item.replace('proj_out', 'proj_attn')
        new_item = new_item.replace('norm', 'group_norm')
        new_item = shave_segments(new_item, n_shave_prefix_segments=
            n_shave_prefix_segments)
        mapping.append({'old': old_item, 'new': new_item})
    return mapping
