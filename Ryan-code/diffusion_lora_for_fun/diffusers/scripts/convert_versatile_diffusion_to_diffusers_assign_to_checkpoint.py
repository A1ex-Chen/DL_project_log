def assign_to_checkpoint(paths, checkpoint, old_checkpoint,
    attention_paths_to_split=None, additional_replacements=None, config=None):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming
    to them. It splits attention layers, and takes into account additional replacements
    that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list
        ), "Paths should be a list of dicts containing 'old' and 'new' keys."
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3
            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else -1
            num_heads = old_tensor.shape[0] // config['num_head_channels'] // 3
            old_tensor = old_tensor.reshape((num_heads, 3 * channels //
                num_heads) + old_tensor.shape[1:])
            query, key, value = old_tensor.split(channels // num_heads, dim=1)
            checkpoint[path_map['query']] = query.reshape(target_shape)
            checkpoint[path_map['key']] = key.reshape(target_shape)
            checkpoint[path_map['value']] = value.reshape(target_shape)
    for path in paths:
        new_path = path['new']
        if (attention_paths_to_split is not None and new_path in
            attention_paths_to_split):
            continue
        new_path = new_path.replace('middle_block.0', 'mid_block.resnets.0')
        new_path = new_path.replace('middle_block.1', 'mid_block.attentions.0')
        new_path = new_path.replace('middle_block.2', 'mid_block.resnets.1')
        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement['old'], replacement
                    ['new'])
        if 'proj_attn.weight' in new_path:
            checkpoint[new_path] = old_checkpoint[path['old']][:, :, 0]
        elif path['old'] in old_checkpoint:
            checkpoint[new_path] = old_checkpoint[path['old']]
