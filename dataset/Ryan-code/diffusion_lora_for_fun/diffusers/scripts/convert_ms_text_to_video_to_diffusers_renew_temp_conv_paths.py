def renew_temp_conv_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        mapping.append({'old': old_item, 'new': old_item})
    return mapping
