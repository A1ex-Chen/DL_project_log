def rename(input_string, max_depth=13):
    string = input_string
    if string.split('.')[0] == 'timestep_embed':
        return string.replace('timestep_embed', 'time_proj')
    depth = 0
    if string.startswith('net.3.'):
        depth += 1
        string = string[6:]
    elif string.startswith('net.'):
        string = string[4:]
    while string.startswith('main.7.'):
        depth += 1
        string = string[7:]
    if string.startswith('main.'):
        string = string[5:]
    if string[:2].isdigit():
        layer_num = string[:2]
        string_left = string[2:]
    else:
        layer_num = string[0]
        string_left = string[1:]
    if depth == max_depth:
        new_layer = MID_NUM_TO_LAYER[layer_num]
        prefix = 'mid_block'
    elif depth > 0 and int(layer_num) < 7:
        new_layer = DOWN_NUM_TO_LAYER[layer_num]
        prefix = f'down_blocks.{depth}'
    elif depth > 0 and int(layer_num) > 7:
        new_layer = UP_NUM_TO_LAYER[layer_num]
        prefix = f'up_blocks.{max_depth - depth - 1}'
    elif depth == 0:
        new_layer = DEPTH_0_TO_LAYER[layer_num]
        prefix = f'up_blocks.{max_depth - 1}' if int(layer_num
            ) > 3 else 'down_blocks.0'
    if not string_left.startswith('.'):
        raise ValueError(
            f'Naming error with {input_string} and string_left: {string_left}.'
            )
    string_left = string_left[1:]
    if 'resnets' in new_layer:
        string_left = convert_resconv_naming(string_left)
    elif 'attentions' in new_layer:
        new_string_left = convert_attn_naming(string_left)
        string_left = new_string_left
    if not isinstance(string_left, list):
        new_string = prefix + '.' + new_layer + '.' + string_left
    else:
        new_string = [(prefix + '.' + new_layer + '.' + s) for s in string_left
            ]
    return new_string
