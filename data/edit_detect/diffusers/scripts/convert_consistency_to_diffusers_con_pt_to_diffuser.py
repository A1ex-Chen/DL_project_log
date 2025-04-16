def con_pt_to_diffuser(checkpoint_path: str, unet_config):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    new_checkpoint = {}
    new_checkpoint['time_embedding.linear_1.weight'] = checkpoint[
        'time_embed.0.weight']
    new_checkpoint['time_embedding.linear_1.bias'] = checkpoint[
        'time_embed.0.bias']
    new_checkpoint['time_embedding.linear_2.weight'] = checkpoint[
        'time_embed.2.weight']
    new_checkpoint['time_embedding.linear_2.bias'] = checkpoint[
        'time_embed.2.bias']
    if unet_config['num_class_embeds'] is not None:
        new_checkpoint['class_embedding.weight'] = checkpoint[
            'label_emb.weight']
    new_checkpoint['conv_in.weight'] = checkpoint['input_blocks.0.0.weight']
    new_checkpoint['conv_in.bias'] = checkpoint['input_blocks.0.0.bias']
    down_block_types = unet_config['down_block_types']
    layers_per_block = unet_config['layers_per_block']
    attention_head_dim = unet_config['attention_head_dim']
    channels_list = unet_config['block_out_channels']
    current_layer = 1
    prev_channels = channels_list[0]
    for i, layer_type in enumerate(down_block_types):
        current_channels = channels_list[i]
        downsample_block_has_skip = current_channels != prev_channels
        if layer_type == 'ResnetDownsampleBlock2D':
            for j in range(layers_per_block):
                new_prefix = f'down_blocks.{i}.resnets.{j}'
                old_prefix = f'input_blocks.{current_layer}.0'
                has_skip = (True if j == 0 and downsample_block_has_skip else
                    False)
                new_checkpoint = convert_resnet(checkpoint, new_checkpoint,
                    old_prefix, new_prefix, has_skip=has_skip)
                current_layer += 1
        elif layer_type == 'AttnDownBlock2D':
            for j in range(layers_per_block):
                new_prefix = f'down_blocks.{i}.resnets.{j}'
                old_prefix = f'input_blocks.{current_layer}.0'
                has_skip = (True if j == 0 and downsample_block_has_skip else
                    False)
                new_checkpoint = convert_resnet(checkpoint, new_checkpoint,
                    old_prefix, new_prefix, has_skip=has_skip)
                new_prefix = f'down_blocks.{i}.attentions.{j}'
                old_prefix = f'input_blocks.{current_layer}.1'
                new_checkpoint = convert_attention(checkpoint,
                    new_checkpoint, old_prefix, new_prefix, attention_head_dim)
                current_layer += 1
        if i != len(down_block_types) - 1:
            new_prefix = f'down_blocks.{i}.downsamplers.0'
            old_prefix = f'input_blocks.{current_layer}.0'
            new_checkpoint = convert_resnet(checkpoint, new_checkpoint,
                old_prefix, new_prefix)
            current_layer += 1
        prev_channels = current_channels
    new_prefix = 'mid_block.resnets.0'
    old_prefix = 'middle_block.0'
    new_checkpoint = convert_resnet(checkpoint, new_checkpoint, old_prefix,
        new_prefix)
    new_prefix = 'mid_block.attentions.0'
    old_prefix = 'middle_block.1'
    new_checkpoint = convert_attention(checkpoint, new_checkpoint,
        old_prefix, new_prefix, attention_head_dim)
    new_prefix = 'mid_block.resnets.1'
    old_prefix = 'middle_block.2'
    new_checkpoint = convert_resnet(checkpoint, new_checkpoint, old_prefix,
        new_prefix)
    current_layer = 0
    up_block_types = unet_config['up_block_types']
    for i, layer_type in enumerate(up_block_types):
        if layer_type == 'ResnetUpsampleBlock2D':
            for j in range(layers_per_block + 1):
                new_prefix = f'up_blocks.{i}.resnets.{j}'
                old_prefix = f'output_blocks.{current_layer}.0'
                new_checkpoint = convert_resnet(checkpoint, new_checkpoint,
                    old_prefix, new_prefix, has_skip=True)
                current_layer += 1
            if i != len(up_block_types) - 1:
                new_prefix = f'up_blocks.{i}.upsamplers.0'
                old_prefix = f'output_blocks.{current_layer - 1}.1'
                new_checkpoint = convert_resnet(checkpoint, new_checkpoint,
                    old_prefix, new_prefix)
        elif layer_type == 'AttnUpBlock2D':
            for j in range(layers_per_block + 1):
                new_prefix = f'up_blocks.{i}.resnets.{j}'
                old_prefix = f'output_blocks.{current_layer}.0'
                new_checkpoint = convert_resnet(checkpoint, new_checkpoint,
                    old_prefix, new_prefix, has_skip=True)
                new_prefix = f'up_blocks.{i}.attentions.{j}'
                old_prefix = f'output_blocks.{current_layer}.1'
                new_checkpoint = convert_attention(checkpoint,
                    new_checkpoint, old_prefix, new_prefix, attention_head_dim)
                current_layer += 1
            if i != len(up_block_types) - 1:
                new_prefix = f'up_blocks.{i}.upsamplers.0'
                old_prefix = f'output_blocks.{current_layer - 1}.2'
                new_checkpoint = convert_resnet(checkpoint, new_checkpoint,
                    old_prefix, new_prefix)
    new_checkpoint['conv_norm_out.weight'] = checkpoint['out.0.weight']
    new_checkpoint['conv_norm_out.bias'] = checkpoint['out.0.bias']
    new_checkpoint['conv_out.weight'] = checkpoint['out.2.weight']
    new_checkpoint['conv_out.bias'] = checkpoint['out.2.bias']
    return new_checkpoint
