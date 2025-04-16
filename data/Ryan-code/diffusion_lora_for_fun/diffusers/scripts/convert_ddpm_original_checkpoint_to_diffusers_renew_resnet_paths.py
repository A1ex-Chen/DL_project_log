def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    mapping = []
    for old_item in old_list:
        new_item = old_item
        new_item = new_item.replace('block.', 'resnets.')
        new_item = new_item.replace('conv_shorcut', 'conv1')
        new_item = new_item.replace('in_shortcut', 'conv_shortcut')
        new_item = new_item.replace('temb_proj', 'time_emb_proj')
        new_item = shave_segments(new_item, n_shave_prefix_segments=
            n_shave_prefix_segments)
        mapping.append({'old': old_item, 'new': new_item})
    return mapping
