def convert_vae_block_state_dict(old_state_dict, prefix_from,
    new_state_dict, prefix_to):
    new_state_dict[f'{prefix_to}.resnets.0.norm1.weight'] = old_state_dict.pop(
        f'{prefix_from}.block.0.norm1.weight')
    new_state_dict[f'{prefix_to}.resnets.0.norm1.bias'] = old_state_dict.pop(
        f'{prefix_from}.block.0.norm1.bias')
    new_state_dict[f'{prefix_to}.resnets.0.conv1.weight'] = old_state_dict.pop(
        f'{prefix_from}.block.0.conv1.weight')
    new_state_dict[f'{prefix_to}.resnets.0.conv1.bias'] = old_state_dict.pop(
        f'{prefix_from}.block.0.conv1.bias')
    new_state_dict[f'{prefix_to}.resnets.0.norm2.weight'] = old_state_dict.pop(
        f'{prefix_from}.block.0.norm2.weight')
    new_state_dict[f'{prefix_to}.resnets.0.norm2.bias'] = old_state_dict.pop(
        f'{prefix_from}.block.0.norm2.bias')
    new_state_dict[f'{prefix_to}.resnets.0.conv2.weight'] = old_state_dict.pop(
        f'{prefix_from}.block.0.conv2.weight')
    new_state_dict[f'{prefix_to}.resnets.0.conv2.bias'] = old_state_dict.pop(
        f'{prefix_from}.block.0.conv2.bias')
    if f'{prefix_from}.block.0.nin_shortcut.weight' in old_state_dict:
        new_state_dict[f'{prefix_to}.resnets.0.conv_shortcut.weight'
            ] = old_state_dict.pop(f'{prefix_from}.block.0.nin_shortcut.weight'
            )
        new_state_dict[f'{prefix_to}.resnets.0.conv_shortcut.bias'
            ] = old_state_dict.pop(f'{prefix_from}.block.0.nin_shortcut.bias')
    new_state_dict[f'{prefix_to}.resnets.1.norm1.weight'] = old_state_dict.pop(
        f'{prefix_from}.block.1.norm1.weight')
    new_state_dict[f'{prefix_to}.resnets.1.norm1.bias'] = old_state_dict.pop(
        f'{prefix_from}.block.1.norm1.bias')
    new_state_dict[f'{prefix_to}.resnets.1.conv1.weight'] = old_state_dict.pop(
        f'{prefix_from}.block.1.conv1.weight')
    new_state_dict[f'{prefix_to}.resnets.1.conv1.bias'] = old_state_dict.pop(
        f'{prefix_from}.block.1.conv1.bias')
    new_state_dict[f'{prefix_to}.resnets.1.norm2.weight'] = old_state_dict.pop(
        f'{prefix_from}.block.1.norm2.weight')
    new_state_dict[f'{prefix_to}.resnets.1.norm2.bias'] = old_state_dict.pop(
        f'{prefix_from}.block.1.norm2.bias')
    new_state_dict[f'{prefix_to}.resnets.1.conv2.weight'] = old_state_dict.pop(
        f'{prefix_from}.block.1.conv2.weight')
    new_state_dict[f'{prefix_to}.resnets.1.conv2.bias'] = old_state_dict.pop(
        f'{prefix_from}.block.1.conv2.bias')
    if f'{prefix_from}.downsample.conv.weight' in old_state_dict:
        new_state_dict[f'{prefix_to}.downsamplers.0.conv.weight'
            ] = old_state_dict.pop(f'{prefix_from}.downsample.conv.weight')
        new_state_dict[f'{prefix_to}.downsamplers.0.conv.bias'
            ] = old_state_dict.pop(f'{prefix_from}.downsample.conv.bias')
    if f'{prefix_from}.upsample.conv.weight' in old_state_dict:
        new_state_dict[f'{prefix_to}.upsamplers.0.conv.weight'
            ] = old_state_dict.pop(f'{prefix_from}.upsample.conv.weight')
        new_state_dict[f'{prefix_to}.upsamplers.0.conv.bias'
            ] = old_state_dict.pop(f'{prefix_from}.upsample.conv.bias')
    if f'{prefix_from}.block.2.norm1.weight' in old_state_dict:
        new_state_dict[f'{prefix_to}.resnets.2.norm1.weight'
            ] = old_state_dict.pop(f'{prefix_from}.block.2.norm1.weight')
        new_state_dict[f'{prefix_to}.resnets.2.norm1.bias'
            ] = old_state_dict.pop(f'{prefix_from}.block.2.norm1.bias')
        new_state_dict[f'{prefix_to}.resnets.2.conv1.weight'
            ] = old_state_dict.pop(f'{prefix_from}.block.2.conv1.weight')
        new_state_dict[f'{prefix_to}.resnets.2.conv1.bias'
            ] = old_state_dict.pop(f'{prefix_from}.block.2.conv1.bias')
        new_state_dict[f'{prefix_to}.resnets.2.norm2.weight'
            ] = old_state_dict.pop(f'{prefix_from}.block.2.norm2.weight')
        new_state_dict[f'{prefix_to}.resnets.2.norm2.bias'
            ] = old_state_dict.pop(f'{prefix_from}.block.2.norm2.bias')
        new_state_dict[f'{prefix_to}.resnets.2.conv2.weight'
            ] = old_state_dict.pop(f'{prefix_from}.block.2.conv2.weight')
        new_state_dict[f'{prefix_to}.resnets.2.conv2.bias'
            ] = old_state_dict.pop(f'{prefix_from}.block.2.conv2.bias')
