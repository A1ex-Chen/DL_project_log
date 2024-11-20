def convert_controlnet_checkpoint(checkpoint, config, **kwargs):
    if 'time_embed.0.weight' in checkpoint:
        controlnet_state_dict = checkpoint
    else:
        controlnet_state_dict = {}
        keys = list(checkpoint.keys())
        controlnet_key = LDM_CONTROLNET_KEY
        for key in keys:
            if key.startswith(controlnet_key):
                controlnet_state_dict[key.replace(controlnet_key, '')
                    ] = checkpoint.get(key)
    new_checkpoint = {}
    ldm_controlnet_keys = DIFFUSERS_TO_LDM_MAPPING['controlnet']['layers']
    for diffusers_key, ldm_key in ldm_controlnet_keys.items():
        if ldm_key not in controlnet_state_dict:
            continue
        new_checkpoint[diffusers_key] = controlnet_state_dict[ldm_key]
    num_input_blocks = len({'.'.join(layer.split('.')[:2]) for layer in
        controlnet_state_dict if 'input_blocks' in layer})
    input_blocks = {layer_id: [key for key in controlnet_state_dict if 
        f'input_blocks.{layer_id}' in key] for layer_id in range(
        num_input_blocks)}
    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config['layers_per_block'] + 1)
        layer_in_block_id = (i - 1) % (config['layers_per_block'] + 1)
        resnets = [key for key in input_blocks[i] if f'input_blocks.{i}.0' in
            key and f'input_blocks.{i}.0.op' not in key]
        update_unet_resnet_ldm_to_diffusers(resnets, new_checkpoint,
            controlnet_state_dict, {'old': f'input_blocks.{i}.0', 'new':
            f'down_blocks.{block_id}.resnets.{layer_in_block_id}'})
        if f'input_blocks.{i}.0.op.weight' in controlnet_state_dict:
            new_checkpoint[f'down_blocks.{block_id}.downsamplers.0.conv.weight'
                ] = controlnet_state_dict.get(f'input_blocks.{i}.0.op.weight')
            new_checkpoint[f'down_blocks.{block_id}.downsamplers.0.conv.bias'
                ] = controlnet_state_dict.get(f'input_blocks.{i}.0.op.bias')
        attentions = [key for key in input_blocks[i] if 
            f'input_blocks.{i}.1' in key]
        if attentions:
            update_unet_attention_ldm_to_diffusers(attentions,
                new_checkpoint, controlnet_state_dict, {'old':
                f'input_blocks.{i}.1', 'new':
                f'down_blocks.{block_id}.attentions.{layer_in_block_id}'})
    for i in range(num_input_blocks):
        new_checkpoint[f'controlnet_down_blocks.{i}.weight'
            ] = controlnet_state_dict.get(f'zero_convs.{i}.0.weight')
        new_checkpoint[f'controlnet_down_blocks.{i}.bias'
            ] = controlnet_state_dict.get(f'zero_convs.{i}.0.bias')
    num_middle_blocks = len({'.'.join(layer.split('.')[:2]) for layer in
        controlnet_state_dict if 'middle_block' in layer})
    middle_blocks = {layer_id: [key for key in controlnet_state_dict if 
        f'middle_block.{layer_id}' in key] for layer_id in range(
        num_middle_blocks)}
    for key in middle_blocks.keys():
        diffusers_key = max(key - 1, 0)
        if key % 2 == 0:
            update_unet_resnet_ldm_to_diffusers(middle_blocks[key],
                new_checkpoint, controlnet_state_dict, mapping={'old':
                f'middle_block.{key}', 'new':
                f'mid_block.resnets.{diffusers_key}'})
        else:
            update_unet_attention_ldm_to_diffusers(middle_blocks[key],
                new_checkpoint, controlnet_state_dict, mapping={'old':
                f'middle_block.{key}', 'new':
                f'mid_block.attentions.{diffusers_key}'})
    new_checkpoint['controlnet_mid_block.weight'] = controlnet_state_dict.get(
        'middle_block_out.0.weight')
    new_checkpoint['controlnet_mid_block.bias'] = controlnet_state_dict.get(
        'middle_block_out.0.bias')
    cond_embedding_blocks = {'.'.join(layer.split('.')[:2]) for layer in
        controlnet_state_dict if 'input_hint_block' in layer and 
        'input_hint_block.0' not in layer and 'input_hint_block.14' not in
        layer}
    num_cond_embedding_blocks = len(cond_embedding_blocks)
    for idx in range(1, num_cond_embedding_blocks + 1):
        diffusers_idx = idx - 1
        cond_block_id = 2 * idx
        new_checkpoint[
            f'controlnet_cond_embedding.blocks.{diffusers_idx}.weight'
            ] = controlnet_state_dict.get(
            f'input_hint_block.{cond_block_id}.weight')
        new_checkpoint[f'controlnet_cond_embedding.blocks.{diffusers_idx}.bias'
            ] = controlnet_state_dict.get(
            f'input_hint_block.{cond_block_id}.bias')
    return new_checkpoint
