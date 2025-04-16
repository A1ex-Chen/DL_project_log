def _maybe_map_sgm_blocks_to_diffusers(state_dict, unet_config, delimiter=
    '_', block_slice_pos=5):
    all_keys = list(state_dict.keys())
    sgm_patterns = ['input_blocks', 'middle_block', 'output_blocks']
    is_in_sgm_format = False
    for key in all_keys:
        if any(p in key for p in sgm_patterns):
            is_in_sgm_format = True
            break
    if not is_in_sgm_format:
        return state_dict
    new_state_dict = {}
    inner_block_map = ['resnets', 'attentions', 'upsamplers']
    input_block_ids, middle_block_ids, output_block_ids = set(), set(), set()
    for layer in all_keys:
        if 'text' in layer:
            new_state_dict[layer] = state_dict.pop(layer)
        else:
            layer_id = int(layer.split(delimiter)[:block_slice_pos][-1])
            if sgm_patterns[0] in layer:
                input_block_ids.add(layer_id)
            elif sgm_patterns[1] in layer:
                middle_block_ids.add(layer_id)
            elif sgm_patterns[2] in layer:
                output_block_ids.add(layer_id)
            else:
                raise ValueError(
                    f'Checkpoint not supported because layer {layer} not supported.'
                    )
    input_blocks = {layer_id: [key for key in state_dict if 
        f'input_blocks{delimiter}{layer_id}' in key] for layer_id in
        input_block_ids}
    middle_blocks = {layer_id: [key for key in state_dict if 
        f'middle_block{delimiter}{layer_id}' in key] for layer_id in
        middle_block_ids}
    output_blocks = {layer_id: [key for key in state_dict if 
        f'output_blocks{delimiter}{layer_id}' in key] for layer_id in
        output_block_ids}
    for i in input_block_ids:
        block_id = (i - 1) // (unet_config.layers_per_block + 1)
        layer_in_block_id = (i - 1) % (unet_config.layers_per_block + 1)
        for key in input_blocks[i]:
            inner_block_id = int(key.split(delimiter)[block_slice_pos])
            inner_block_key = inner_block_map[inner_block_id
                ] if 'op' not in key else 'downsamplers'
            inner_layers_in_block = str(layer_in_block_id
                ) if 'op' not in key else '0'
            new_key = delimiter.join(key.split(delimiter)[:block_slice_pos -
                1] + [str(block_id), inner_block_key, inner_layers_in_block
                ] + key.split(delimiter)[block_slice_pos + 1:])
            new_state_dict[new_key] = state_dict.pop(key)
    for i in middle_block_ids:
        key_part = None
        if i == 0:
            key_part = [inner_block_map[0], '0']
        elif i == 1:
            key_part = [inner_block_map[1], '0']
        elif i == 2:
            key_part = [inner_block_map[0], '1']
        else:
            raise ValueError(f'Invalid middle block id {i}.')
        for key in middle_blocks[i]:
            new_key = delimiter.join(key.split(delimiter)[:block_slice_pos -
                1] + key_part + key.split(delimiter)[block_slice_pos:])
            new_state_dict[new_key] = state_dict.pop(key)
    for i in output_block_ids:
        block_id = i // (unet_config.layers_per_block + 1)
        layer_in_block_id = i % (unet_config.layers_per_block + 1)
        for key in output_blocks[i]:
            inner_block_id = int(key.split(delimiter)[block_slice_pos])
            inner_block_key = inner_block_map[inner_block_id]
            inner_layers_in_block = str(layer_in_block_id
                ) if inner_block_id < 2 else '0'
            new_key = delimiter.join(key.split(delimiter)[:block_slice_pos -
                1] + [str(block_id), inner_block_key, inner_layers_in_block
                ] + key.split(delimiter)[block_slice_pos + 1:])
            new_state_dict[new_key] = state_dict.pop(key)
    if len(state_dict) > 0:
        raise ValueError(
            'At this point all state dict entries have to be converted.')
    return new_state_dict
