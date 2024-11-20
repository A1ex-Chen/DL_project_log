def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item
        new_item = new_item.replace('time_stack', 'temporal_transformer_blocks'
            )
        new_item = new_item.replace('time_pos_embed.0.bias',
            'time_pos_embed.linear_1.bias')
        new_item = new_item.replace('time_pos_embed.0.weight',
            'time_pos_embed.linear_1.weight')
        new_item = new_item.replace('time_pos_embed.2.bias',
            'time_pos_embed.linear_2.bias')
        new_item = new_item.replace('time_pos_embed.2.weight',
            'time_pos_embed.linear_2.weight')
        mapping.append({'old': old_item, 'new': new_item})
    return mapping
