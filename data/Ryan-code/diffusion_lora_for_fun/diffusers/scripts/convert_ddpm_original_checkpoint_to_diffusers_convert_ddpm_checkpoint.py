def convert_ddpm_checkpoint(checkpoint, config):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """
    new_checkpoint = {}
    new_checkpoint['time_embedding.linear_1.weight'] = checkpoint[
        'temb.dense.0.weight']
    new_checkpoint['time_embedding.linear_1.bias'] = checkpoint[
        'temb.dense.0.bias']
    new_checkpoint['time_embedding.linear_2.weight'] = checkpoint[
        'temb.dense.1.weight']
    new_checkpoint['time_embedding.linear_2.bias'] = checkpoint[
        'temb.dense.1.bias']
    new_checkpoint['conv_norm_out.weight'] = checkpoint['norm_out.weight']
    new_checkpoint['conv_norm_out.bias'] = checkpoint['norm_out.bias']
    new_checkpoint['conv_in.weight'] = checkpoint['conv_in.weight']
    new_checkpoint['conv_in.bias'] = checkpoint['conv_in.bias']
    new_checkpoint['conv_out.weight'] = checkpoint['conv_out.weight']
    new_checkpoint['conv_out.bias'] = checkpoint['conv_out.bias']
    num_down_blocks = len({'.'.join(layer.split('.')[:2]) for layer in
        checkpoint if 'down' in layer})
    down_blocks = {layer_id: [key for key in checkpoint if 
        f'down.{layer_id}' in key] for layer_id in range(num_down_blocks)}
    num_up_blocks = len({'.'.join(layer.split('.')[:2]) for layer in
        checkpoint if 'up' in layer})
    up_blocks = {layer_id: [key for key in checkpoint if f'up.{layer_id}' in
        key] for layer_id in range(num_up_blocks)}
    for i in range(num_down_blocks):
        block_id = (i - 1) // (config['layers_per_block'] + 1)
        if any('downsample' in layer for layer in down_blocks[i]):
            new_checkpoint[f'down_blocks.{i}.downsamplers.0.conv.weight'
                ] = checkpoint[f'down.{i}.downsample.op.weight']
            new_checkpoint[f'down_blocks.{i}.downsamplers.0.conv.bias'
                ] = checkpoint[f'down.{i}.downsample.op.bias']
        if any('block' in layer for layer in down_blocks[i]):
            num_blocks = len({'.'.join(shave_segments(layer, 2).split('.')[
                :2]) for layer in down_blocks[i] if 'block' in layer})
            blocks = {layer_id: [key for key in down_blocks[i] if 
                f'block.{layer_id}' in key] for layer_id in range(num_blocks)}
            if num_blocks > 0:
                for j in range(config['layers_per_block']):
                    paths = renew_resnet_paths(blocks[j])
                    assign_to_checkpoint(paths, new_checkpoint, checkpoint)
        if any('attn' in layer for layer in down_blocks[i]):
            num_attn = len({'.'.join(shave_segments(layer, 2).split('.')[:2
                ]) for layer in down_blocks[i] if 'attn' in layer})
            attns = {layer_id: [key for key in down_blocks[i] if 
                f'attn.{layer_id}' in key] for layer_id in range(num_blocks)}
            if num_attn > 0:
                for j in range(config['layers_per_block']):
                    paths = renew_attention_paths(attns[j])
                    assign_to_checkpoint(paths, new_checkpoint, checkpoint,
                        config=config)
    mid_block_1_layers = [key for key in checkpoint if 'mid.block_1' in key]
    mid_block_2_layers = [key for key in checkpoint if 'mid.block_2' in key]
    mid_attn_1_layers = [key for key in checkpoint if 'mid.attn_1' in key]
    paths = renew_resnet_paths(mid_block_1_layers)
    assign_to_checkpoint(paths, new_checkpoint, checkpoint,
        additional_replacements=[{'old': 'mid.', 'new': 'mid_new_2.'}, {
        'old': 'block_1', 'new': 'resnets.0'}])
    paths = renew_resnet_paths(mid_block_2_layers)
    assign_to_checkpoint(paths, new_checkpoint, checkpoint,
        additional_replacements=[{'old': 'mid.', 'new': 'mid_new_2.'}, {
        'old': 'block_2', 'new': 'resnets.1'}])
    paths = renew_attention_paths(mid_attn_1_layers, in_mid=True)
    assign_to_checkpoint(paths, new_checkpoint, checkpoint,
        additional_replacements=[{'old': 'mid.', 'new': 'mid_new_2.'}, {
        'old': 'attn_1', 'new': 'attentions.0'}])
    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        if any('upsample' in layer for layer in up_blocks[i]):
            new_checkpoint[f'up_blocks.{block_id}.upsamplers.0.conv.weight'
                ] = checkpoint[f'up.{i}.upsample.conv.weight']
            new_checkpoint[f'up_blocks.{block_id}.upsamplers.0.conv.bias'
                ] = checkpoint[f'up.{i}.upsample.conv.bias']
        if any('block' in layer for layer in up_blocks[i]):
            num_blocks = len({'.'.join(shave_segments(layer, 2).split('.')[
                :2]) for layer in up_blocks[i] if 'block' in layer})
            blocks = {layer_id: [key for key in up_blocks[i] if 
                f'block.{layer_id}' in key] for layer_id in range(num_blocks)}
            if num_blocks > 0:
                for j in range(config['layers_per_block'] + 1):
                    replace_indices = {'old': f'up_blocks.{i}', 'new':
                        f'up_blocks.{block_id}'}
                    paths = renew_resnet_paths(blocks[j])
                    assign_to_checkpoint(paths, new_checkpoint, checkpoint,
                        additional_replacements=[replace_indices])
        if any('attn' in layer for layer in up_blocks[i]):
            num_attn = len({'.'.join(shave_segments(layer, 2).split('.')[:2
                ]) for layer in up_blocks[i] if 'attn' in layer})
            attns = {layer_id: [key for key in up_blocks[i] if 
                f'attn.{layer_id}' in key] for layer_id in range(num_blocks)}
            if num_attn > 0:
                for j in range(config['layers_per_block'] + 1):
                    replace_indices = {'old': f'up_blocks.{i}', 'new':
                        f'up_blocks.{block_id}'}
                    paths = renew_attention_paths(attns[j])
                    assign_to_checkpoint(paths, new_checkpoint, checkpoint,
                        additional_replacements=[replace_indices])
    new_checkpoint = {k.replace('mid_new_2', 'mid_block'): v for k, v in
        new_checkpoint.items()}
    return new_checkpoint
