def convert_ldm_unet_checkpoint(checkpoint, config, path=None, extract_ema=
    False, controlnet=False):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """
    unet_state_dict = {}
    keys = list(checkpoint.keys())
    if controlnet:
        unet_key = 'control_model.'
    else:
        unet_key = 'model.diffusion_model.'
    if sum(k.startswith('model_ema') for k in keys) > 100 and extract_ema:
        print(f'Checkpoint {path} has both EMA and non-EMA weights.')
        print(
            'In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag.'
            )
        for key in keys:
            if key.startswith('model.diffusion_model'):
                flat_ema_key = 'model_ema.' + ''.join(key.split('.')[1:])
                unet_state_dict[key.replace(unet_key, '')] = checkpoint.pop(
                    flat_ema_key)
    else:
        if sum(k.startswith('model_ema') for k in keys) > 100:
            print(
                'In this conversion only the non-EMA weights are extracted. If you want to instead extract the EMA weights (usually better for inference), please make sure to add the `--extract_ema` flag.'
                )
        for key in keys:
            if key.startswith(unet_key):
                unet_state_dict[key.replace(unet_key, '')] = checkpoint.pop(key
                    )
    new_checkpoint = {}
    new_checkpoint['time_embedding.linear_1.weight'] = unet_state_dict[
        'time_embed.0.weight']
    new_checkpoint['time_embedding.linear_1.bias'] = unet_state_dict[
        'time_embed.0.bias']
    new_checkpoint['time_embedding.linear_2.weight'] = unet_state_dict[
        'time_embed.2.weight']
    new_checkpoint['time_embedding.linear_2.bias'] = unet_state_dict[
        'time_embed.2.bias']
    if config['class_embed_type'] is None:
        ...
    elif config['class_embed_type'] == 'timestep' or config['class_embed_type'
        ] == 'projection':
        new_checkpoint['class_embedding.linear_1.weight'] = unet_state_dict[
            'label_emb.0.0.weight']
        new_checkpoint['class_embedding.linear_1.bias'] = unet_state_dict[
            'label_emb.0.0.bias']
        new_checkpoint['class_embedding.linear_2.weight'] = unet_state_dict[
            'label_emb.0.2.weight']
        new_checkpoint['class_embedding.linear_2.bias'] = unet_state_dict[
            'label_emb.0.2.bias']
    else:
        raise NotImplementedError(
            f"Not implemented `class_embed_type`: {config['class_embed_type']}"
            )
    new_checkpoint['conv_in.weight'] = unet_state_dict[
        'input_blocks.0.0.weight']
    new_checkpoint['conv_in.bias'] = unet_state_dict['input_blocks.0.0.bias']
    if not controlnet:
        new_checkpoint['conv_norm_out.weight'] = unet_state_dict['out.0.weight'
            ]
        new_checkpoint['conv_norm_out.bias'] = unet_state_dict['out.0.bias']
        new_checkpoint['conv_out.weight'] = unet_state_dict['out.2.weight']
        new_checkpoint['conv_out.bias'] = unet_state_dict['out.2.bias']
    num_input_blocks = len({'.'.join(layer.split('.')[:2]) for layer in
        unet_state_dict if 'input_blocks' in layer})
    input_blocks = {layer_id: [key for key in unet_state_dict if 
        f'input_blocks.{layer_id}' in key] for layer_id in range(
        num_input_blocks)}
    num_middle_blocks = len({'.'.join(layer.split('.')[:2]) for layer in
        unet_state_dict if 'middle_block' in layer})
    middle_blocks = {layer_id: [key for key in unet_state_dict if 
        f'middle_block.{layer_id}' in key] for layer_id in range(
        num_middle_blocks)}
    num_output_blocks = len({'.'.join(layer.split('.')[:2]) for layer in
        unet_state_dict if 'output_blocks' in layer})
    output_blocks = {layer_id: [key for key in unet_state_dict if 
        f'output_blocks.{layer_id}' in key] for layer_id in range(
        num_output_blocks)}
    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config['layers_per_block'] + 1)
        layer_in_block_id = (i - 1) % (config['layers_per_block'] + 1)
        resnets = [key for key in input_blocks[i] if f'input_blocks.{i}.0' in
            key and f'input_blocks.{i}.0.op' not in key]
        attentions = [key for key in input_blocks[i] if 
            f'input_blocks.{i}.1' in key]
        if f'input_blocks.{i}.0.op.weight' in unet_state_dict:
            new_checkpoint[f'down_blocks.{block_id}.downsamplers.0.conv.weight'
                ] = unet_state_dict.pop(f'input_blocks.{i}.0.op.weight')
            new_checkpoint[f'down_blocks.{block_id}.downsamplers.0.conv.bias'
                ] = unet_state_dict.pop(f'input_blocks.{i}.0.op.bias')
        paths = renew_resnet_paths(resnets)
        meta_path = {'old': f'input_blocks.{i}.0', 'new':
            f'down_blocks.{block_id}.resnets.{layer_in_block_id}'}
        assign_to_checkpoint(paths, new_checkpoint, unet_state_dict,
            additional_replacements=[meta_path], config=config)
        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {'old': f'input_blocks.{i}.1', 'new':
                f'down_blocks.{block_id}.attentions.{layer_in_block_id}'}
            assign_to_checkpoint(paths, new_checkpoint, unet_state_dict,
                additional_replacements=[meta_path], config=config)
    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]
    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict,
        config=config)
    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict,
        config=config)
    attentions_paths = renew_attention_paths(attentions)
    meta_path = {'old': 'middle_block.1', 'new': 'mid_block.attentions.0'}
    assign_to_checkpoint(attentions_paths, new_checkpoint, unet_state_dict,
        additional_replacements=[meta_path], config=config)
    for i in range(num_output_blocks):
        block_id = i // (config['layers_per_block'] + 1)
        layer_in_block_id = i % (config['layers_per_block'] + 1)
        output_block_layers = [shave_segments(name, 2) for name in
            output_blocks[i]]
        output_block_list = {}
        for layer in output_block_layers:
            layer_id, layer_name = layer.split('.')[0], shave_segments(layer, 1
                )
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]
        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if 
                f'output_blocks.{i}.0' in key]
            attentions = [key for key in output_blocks[i] if 
                f'output_blocks.{i}.1' in key]
            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)
            meta_path = {'old': f'output_blocks.{i}.0', 'new':
                f'up_blocks.{block_id}.resnets.{layer_in_block_id}'}
            assign_to_checkpoint(paths, new_checkpoint, unet_state_dict,
                additional_replacements=[meta_path], config=config)
            output_block_list = {k: sorted(v) for k, v in output_block_list
                .items()}
            if ['conv.bias', 'conv.weight'] in output_block_list.values():
                index = list(output_block_list.values()).index(['conv.bias',
                    'conv.weight'])
                new_checkpoint[f'up_blocks.{block_id}.upsamplers.0.conv.weight'
                    ] = unet_state_dict[
                    f'output_blocks.{i}.{index}.conv.weight']
                new_checkpoint[f'up_blocks.{block_id}.upsamplers.0.conv.bias'
                    ] = unet_state_dict[f'output_blocks.{i}.{index}.conv.bias']
                if len(attentions) == 2:
                    attentions = []
            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {'old': f'output_blocks.{i}.1', 'new':
                    f'up_blocks.{block_id}.attentions.{layer_in_block_id}'}
                assign_to_checkpoint(paths, new_checkpoint, unet_state_dict,
                    additional_replacements=[meta_path], config=config)
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers,
                n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = '.'.join(['output_blocks', str(i), path['old']])
                new_path = '.'.join(['up_blocks', str(block_id), 'resnets',
                    str(layer_in_block_id), path['new']])
                new_checkpoint[new_path] = unet_state_dict[old_path]
    if controlnet:
        orig_index = 0
        new_checkpoint['controlnet_cond_embedding.conv_in.weight'
            ] = unet_state_dict.pop(f'input_hint_block.{orig_index}.weight')
        new_checkpoint['controlnet_cond_embedding.conv_in.bias'
            ] = unet_state_dict.pop(f'input_hint_block.{orig_index}.bias')
        orig_index += 2
        diffusers_index = 0
        while diffusers_index < 6:
            new_checkpoint[
                f'controlnet_cond_embedding.blocks.{diffusers_index}.weight'
                ] = unet_state_dict.pop(f'input_hint_block.{orig_index}.weight'
                )
            new_checkpoint[
                f'controlnet_cond_embedding.blocks.{diffusers_index}.bias'
                ] = unet_state_dict.pop(f'input_hint_block.{orig_index}.bias')
            diffusers_index += 1
            orig_index += 2
        new_checkpoint['controlnet_cond_embedding.conv_out.weight'
            ] = unet_state_dict.pop(f'input_hint_block.{orig_index}.weight')
        new_checkpoint['controlnet_cond_embedding.conv_out.bias'
            ] = unet_state_dict.pop(f'input_hint_block.{orig_index}.bias')
        for i in range(num_input_blocks):
            new_checkpoint[f'controlnet_down_blocks.{i}.weight'
                ] = unet_state_dict.pop(f'zero_convs.{i}.0.weight')
            new_checkpoint[f'controlnet_down_blocks.{i}.bias'
                ] = unet_state_dict.pop(f'zero_convs.{i}.0.bias')
        new_checkpoint['controlnet_mid_block.weight'] = unet_state_dict.pop(
            'middle_block_out.0.weight')
        new_checkpoint['controlnet_mid_block.bias'] = unet_state_dict.pop(
            'middle_block_out.0.bias')
    return new_checkpoint
