def assign_to_checkpoint(paths, checkpoint, old_checkpoint,
    attention_paths_to_split=None, additional_replacements=None, config=None):
    assert isinstance(paths, list
        ), "Paths should be a list of dicts containing 'old' and 'new' keys."
    if attention_paths_to_split is not None:
        if config is None:
            raise ValueError(
                "Please specify the config if setting 'attention_paths_to_split' to 'True'."
                )
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3
            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else -1
            num_heads = old_tensor.shape[0] // config.get('num_head_channels',
                1) // 3
            old_tensor = old_tensor.reshape((num_heads, 3 * channels //
                num_heads) + old_tensor.shape[1:])
            query, key, value = old_tensor.split(channels // num_heads, dim=1)
            checkpoint[path_map['query']] = query.reshape(target_shape
                ).squeeze()
            checkpoint[path_map['key']] = key.reshape(target_shape).squeeze()
            checkpoint[path_map['value']] = value.reshape(target_shape
                ).squeeze()
    for path in paths:
        new_path = path['new']
        if (attention_paths_to_split is not None and new_path in
            attention_paths_to_split):
            continue
        new_path = new_path.replace('down.', 'down_blocks.')
        new_path = new_path.replace('up.', 'up_blocks.')
        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement['old'], replacement
                    ['new'])
        if 'attentions' in new_path:
            checkpoint[new_path] = old_checkpoint[path['old']].squeeze()
        else:
            checkpoint[new_path] = old_checkpoint[path['old']]
