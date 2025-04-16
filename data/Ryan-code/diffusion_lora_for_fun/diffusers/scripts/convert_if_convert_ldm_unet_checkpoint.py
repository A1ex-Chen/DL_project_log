def convert_ldm_unet_checkpoint(unet_state_dict, config, path=None):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """
    new_checkpoint = {}
    new_checkpoint['time_embedding.linear_1.weight'] = unet_state_dict[
        'time_embed.0.weight']
    new_checkpoint['time_embedding.linear_1.bias'] = unet_state_dict[
        'time_embed.0.bias']
    new_checkpoint['time_embedding.linear_2.weight'] = unet_state_dict[
        'time_embed.2.weight']
    new_checkpoint['time_embedding.linear_2.bias'] = unet_state_dict[
        'time_embed.2.bias']
    if config['class_embed_type'] in [None, 'identity']:
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
    new_checkpoint['conv_norm_out.weight'] = unet_state_dict['out.0.weight']
    new_checkpoint['conv_norm_out.bias'] = unet_state_dict['out.0.bias']
    new_checkpoint['conv_out.weight'] = unet_state_dict['out.2.weight']
    new_checkpoint['conv_out.bias'] = unet_state_dict['out.2.bias']
    num_input_blocks = len({'.'.join(layer.split('.')[:2]) for layer in
        unet_state_dict if 'input_blocks' in layer})
    input_blocks = {layer_id: [key for key in unet_state_dict if 
        f'input_blocks.{layer_id}.' in key] for layer_id in range(
        num_input_blocks)}
    num_middle_blocks = len({'.'.join(layer.split('.')[:2]) for layer in
        unet_state_dict if 'middle_block' in layer})
    middle_blocks = {layer_id: [key for key in unet_state_dict if 
        f'middle_block.{layer_id}' in key] for layer_id in range(
        num_middle_blocks)}
    num_output_blocks = len({'.'.join(layer.split('.')[:2]) for layer in
        unet_state_dict if 'output_blocks' in layer})
    output_blocks = {layer_id: [key for key in unet_state_dict if 
        f'output_blocks.{layer_id}.' in key] for layer_id in range(
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
        block_type = config['down_block_types'][block_id]
        if (block_type == 'ResnetDownsampleBlock2D' or block_type ==
            'SimpleCrossAttnDownBlock2D') and i in [4, 8, 12, 16]:
            meta_path = {'old': f'input_blocks.{i}.0', 'new':
                f'down_blocks.{block_id}.downsamplers.0'}
        else:
            meta_path = {'old': f'input_blocks.{i}.0', 'new':
                f'down_blocks.{block_id}.resnets.{layer_in_block_id}'}
        assign_to_checkpoint(paths, new_checkpoint, unet_state_dict,
            additional_replacements=[meta_path], config=config)
        if len(attentions):
            old_path = f'input_blocks.{i}.1'
            new_path = f'down_blocks.{block_id}.attentions.{layer_in_block_id}'
            assign_attention_to_checkpoint(new_checkpoint=new_checkpoint,
                unet_state_dict=unet_state_dict, old_path=old_path,
                new_path=new_path, config=config)
            paths = renew_attention_paths(attentions)
            meta_path = {'old': old_path, 'new': new_path}
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
    old_path = 'middle_block.1'
    new_path = 'mid_block.attentions.0'
    assign_attention_to_checkpoint(new_checkpoint=new_checkpoint,
        unet_state_dict=unet_state_dict, old_path=old_path, new_path=
        new_path, config=config)
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
                old_path = f'output_blocks.{i}.1'
                new_path = (
                    f'up_blocks.{block_id}.attentions.{layer_in_block_id}')
                assign_attention_to_checkpoint(new_checkpoint=
                    new_checkpoint, unet_state_dict=unet_state_dict,
                    old_path=old_path, new_path=new_path, config=config)
                paths = renew_attention_paths(attentions)
                meta_path = {'old': old_path, 'new': new_path}
                assign_to_checkpoint(paths, new_checkpoint, unet_state_dict,
                    additional_replacements=[meta_path], config=config)
            if len(output_block_list) == 3:
                resnets = [key for key in output_blocks[i] if 
                    f'output_blocks.{i}.2' in key]
                paths = renew_resnet_paths(resnets)
                meta_path = {'old': f'output_blocks.{i}.2', 'new':
                    f'up_blocks.{block_id}.upsamplers.0'}
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
    if 'encoder_proj.weight' in unet_state_dict:
        new_checkpoint['encoder_hid_proj.weight'] = unet_state_dict.pop(
            'encoder_proj.weight')
        new_checkpoint['encoder_hid_proj.bias'] = unet_state_dict.pop(
            'encoder_proj.bias')
    if 'encoder_pooling.0.weight' in unet_state_dict:
        new_checkpoint['add_embedding.norm1.weight'] = unet_state_dict.pop(
            'encoder_pooling.0.weight')
        new_checkpoint['add_embedding.norm1.bias'] = unet_state_dict.pop(
            'encoder_pooling.0.bias')
        new_checkpoint['add_embedding.pool.positional_embedding'
            ] = unet_state_dict.pop('encoder_pooling.1.positional_embedding')
        new_checkpoint['add_embedding.pool.k_proj.weight'
            ] = unet_state_dict.pop('encoder_pooling.1.k_proj.weight')
        new_checkpoint['add_embedding.pool.k_proj.bias'] = unet_state_dict.pop(
            'encoder_pooling.1.k_proj.bias')
        new_checkpoint['add_embedding.pool.q_proj.weight'
            ] = unet_state_dict.pop('encoder_pooling.1.q_proj.weight')
        new_checkpoint['add_embedding.pool.q_proj.bias'] = unet_state_dict.pop(
            'encoder_pooling.1.q_proj.bias')
        new_checkpoint['add_embedding.pool.v_proj.weight'
            ] = unet_state_dict.pop('encoder_pooling.1.v_proj.weight')
        new_checkpoint['add_embedding.pool.v_proj.bias'] = unet_state_dict.pop(
            'encoder_pooling.1.v_proj.bias')
        new_checkpoint['add_embedding.proj.weight'] = unet_state_dict.pop(
            'encoder_pooling.2.weight')
        new_checkpoint['add_embedding.proj.bias'] = unet_state_dict.pop(
            'encoder_pooling.2.bias')
        new_checkpoint['add_embedding.norm2.weight'] = unet_state_dict.pop(
            'encoder_pooling.3.weight')
        new_checkpoint['add_embedding.norm2.bias'] = unet_state_dict.pop(
            'encoder_pooling.3.bias')
    return new_checkpoint
