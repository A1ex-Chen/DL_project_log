def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0, is_temporal
    =False):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item
        new_item = old_item.replace('in_layers.0', 'norm1')
        new_item = new_item.replace('in_layers.2', 'conv1')
        new_item = new_item.replace('out_layers.0', 'norm2')
        new_item = new_item.replace('out_layers.3', 'conv2')
        new_item = new_item.replace('skip_connection', 'conv_shortcut')
        new_item = new_item.replace('time_stack.', 'temporal_res_block.')
        new_item = new_item.replace('conv1', 'spatial_res_block.conv1')
        new_item = new_item.replace('norm1', 'spatial_res_block.norm1')
        new_item = new_item.replace('conv2', 'spatial_res_block.conv2')
        new_item = new_item.replace('norm2', 'spatial_res_block.norm2')
        new_item = new_item.replace('nin_shortcut',
            'spatial_res_block.conv_shortcut')
        new_item = new_item.replace('mix_factor',
            'spatial_res_block.time_mixer.mix_factor')
        new_item = shave_segments(new_item, n_shave_prefix_segments=
            n_shave_prefix_segments)
        mapping.append({'old': old_item, 'new': new_item})
    return mapping
