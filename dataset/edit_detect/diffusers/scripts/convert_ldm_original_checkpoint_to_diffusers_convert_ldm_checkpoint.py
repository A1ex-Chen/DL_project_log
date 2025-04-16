def convert_ldm_checkpoint(checkpoint, config):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """
    new_checkpoint = {}
    new_checkpoint['time_embedding.linear_1.weight'] = checkpoint[
        'time_embed.0.weight']
    new_checkpoint['time_embedding.linear_1.bias'] = checkpoint[
        'time_embed.0.bias']
    new_checkpoint['time_embedding.linear_2.weight'] = checkpoint[
        'time_embed.2.weight']
    new_checkpoint['time_embedding.linear_2.bias'] = checkpoint[
        'time_embed.2.bias']
    new_checkpoint['conv_in.weight'] = checkpoint['input_blocks.0.0.weight']
    new_checkpoint['conv_in.bias'] = checkpoint['input_blocks.0.0.bias']
    new_checkpoint['conv_norm_out.weight'] = checkpoint['out.0.weight']
    new_checkpoint['conv_norm_out.bias'] = checkpoint['out.0.bias']
    new_checkpoint['conv_out.weight'] = checkpoint['out.2.weight']
    new_checkpoint['conv_out.bias'] = checkpoint['out.2.bias']
    num_input_blocks = len({'.'.join(layer.split('.')[:2]) for layer in
        checkpoint if 'input_blocks' in layer})
    input_blocks = {layer_id: [key for key in checkpoint if 
        f'input_blocks.{layer_id}' in key] for layer_id in range(
        num_input_blocks)}
    num_middle_blocks = len({'.'.join(layer.split('.')[:2]) for layer in
        checkpoint if 'middle_block' in layer})
    middle_blocks = {layer_id: [key for key in checkpoint if 
        f'middle_block.{layer_id}' in key] for layer_id in range(
        num_middle_blocks)}
    num_output_blocks = len({'.'.join(layer.split('.')[:2]) for layer in
        checkpoint if 'output_blocks' in layer})
    output_blocks = {layer_id: [key for key in checkpoint if 
        f'output_blocks.{layer_id}' in key] for layer_id in range(
        num_output_blocks)}
    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config['num_res_blocks'] + 1)
        layer_in_block_id = (i - 1) % (config['num_res_blocks'] + 1)
        resnets = [key for key in input_blocks[i] if f'input_blocks.{i}.0' in
            key]
        attentions = [key for key in input_blocks[i] if 
            f'input_blocks.{i}.1' in key]
        if f'input_blocks.{i}.0.op.weight' in checkpoint:
            new_checkpoint[f'down_blocks.{block_id}.downsamplers.0.conv.weight'
                ] = checkpoint[f'input_blocks.{i}.0.op.weight']
            new_checkpoint[f'down_blocks.{block_id}.downsamplers.0.conv.bias'
                ] = checkpoint[f'input_blocks.{i}.0.op.bias']
            continue
        paths = renew_resnet_paths(resnets)
        meta_path = {'old': f'input_blocks.{i}.0', 'new':
            f'down_blocks.{block_id}.resnets.{layer_in_block_id}'}
        resnet_op = {'old': 'resnets.2.op', 'new': 'downsamplers.0.op'}
        assign_to_checkpoint(paths, new_checkpoint, checkpoint,
            additional_replacements=[meta_path, resnet_op], config=config)
        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {'old': f'input_blocks.{i}.1', 'new':
                f'down_blocks.{block_id}.attentions.{layer_in_block_id}'}
            to_split = {f'input_blocks.{i}.1.qkv.bias': {'key':
                f'down_blocks.{block_id}.attentions.{layer_in_block_id}.key.bias'
                , 'query':
                f'down_blocks.{block_id}.attentions.{layer_in_block_id}.query.bias'
                , 'value':
                f'down_blocks.{block_id}.attentions.{layer_in_block_id}.value.bias'
                }, f'input_blocks.{i}.1.qkv.weight': {'key':
                f'down_blocks.{block_id}.attentions.{layer_in_block_id}.key.weight'
                , 'query':
                f'down_blocks.{block_id}.attentions.{layer_in_block_id}.query.weight'
                , 'value':
                f'down_blocks.{block_id}.attentions.{layer_in_block_id}.value.weight'
                }}
            assign_to_checkpoint(paths, new_checkpoint, checkpoint,
                additional_replacements=[meta_path],
                attention_paths_to_split=to_split, config=config)
    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]
    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, checkpoint, config
        =config)
    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, checkpoint, config
        =config)
    attentions_paths = renew_attention_paths(attentions)
    to_split = {'middle_block.1.qkv.bias': {'key':
        'mid_block.attentions.0.key.bias', 'query':
        'mid_block.attentions.0.query.bias', 'value':
        'mid_block.attentions.0.value.bias'}, 'middle_block.1.qkv.weight':
        {'key': 'mid_block.attentions.0.key.weight', 'query':
        'mid_block.attentions.0.query.weight', 'value':
        'mid_block.attentions.0.value.weight'}}
    assign_to_checkpoint(attentions_paths, new_checkpoint, checkpoint,
        attention_paths_to_split=to_split, config=config)
    for i in range(num_output_blocks):
        block_id = i // (config['num_res_blocks'] + 1)
        layer_in_block_id = i % (config['num_res_blocks'] + 1)
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
            assign_to_checkpoint(paths, new_checkpoint, checkpoint,
                additional_replacements=[meta_path], config=config)
            if ['conv.weight', 'conv.bias'] in output_block_list.values():
                index = list(output_block_list.values()).index([
                    'conv.weight', 'conv.bias'])
                new_checkpoint[f'up_blocks.{block_id}.upsamplers.0.conv.weight'
                    ] = checkpoint[f'output_blocks.{i}.{index}.conv.weight']
                new_checkpoint[f'up_blocks.{block_id}.upsamplers.0.conv.bias'
                    ] = checkpoint[f'output_blocks.{i}.{index}.conv.bias']
                if len(attentions) == 2:
                    attentions = []
            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {'old': f'output_blocks.{i}.1', 'new':
                    f'up_blocks.{block_id}.attentions.{layer_in_block_id}'}
                to_split = {f'output_blocks.{i}.1.qkv.bias': {'key':
                    f'up_blocks.{block_id}.attentions.{layer_in_block_id}.key.bias'
                    , 'query':
                    f'up_blocks.{block_id}.attentions.{layer_in_block_id}.query.bias'
                    , 'value':
                    f'up_blocks.{block_id}.attentions.{layer_in_block_id}.value.bias'
                    }, f'output_blocks.{i}.1.qkv.weight': {'key':
                    f'up_blocks.{block_id}.attentions.{layer_in_block_id}.key.weight'
                    , 'query':
                    f'up_blocks.{block_id}.attentions.{layer_in_block_id}.query.weight'
                    , 'value':
                    f'up_blocks.{block_id}.attentions.{layer_in_block_id}.value.weight'
                    }}
                assign_to_checkpoint(paths, new_checkpoint, checkpoint,
                    additional_replacements=[meta_path],
                    attention_paths_to_split=to_split if any('qkv' in key for
                    key in attentions) else None, config=config)
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers,
                n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = '.'.join(['output_blocks', str(i), path['old']])
                new_path = '.'.join(['up_blocks', str(block_id), 'resnets',
                    str(layer_in_block_id), path['new']])
                new_checkpoint[new_path] = checkpoint[old_path]
    return new_checkpoint
