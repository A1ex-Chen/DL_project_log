def custom_convert_ldm_vae_checkpoint(checkpoint, config):
    vae_state_dict = checkpoint
    new_checkpoint = {}
    new_checkpoint['encoder.conv_in.weight'] = vae_state_dict[
        'encoder.conv_in.weight']
    new_checkpoint['encoder.conv_in.bias'] = vae_state_dict[
        'encoder.conv_in.bias']
    new_checkpoint['encoder.conv_out.weight'] = vae_state_dict[
        'encoder.conv_out.weight']
    new_checkpoint['encoder.conv_out.bias'] = vae_state_dict[
        'encoder.conv_out.bias']
    new_checkpoint['encoder.conv_norm_out.weight'] = vae_state_dict[
        'encoder.norm_out.weight']
    new_checkpoint['encoder.conv_norm_out.bias'] = vae_state_dict[
        'encoder.norm_out.bias']
    new_checkpoint['decoder.conv_in.weight'] = vae_state_dict[
        'decoder.conv_in.weight']
    new_checkpoint['decoder.conv_in.bias'] = vae_state_dict[
        'decoder.conv_in.bias']
    new_checkpoint['decoder.conv_out.weight'] = vae_state_dict[
        'decoder.conv_out.weight']
    new_checkpoint['decoder.conv_out.bias'] = vae_state_dict[
        'decoder.conv_out.bias']
    new_checkpoint['decoder.conv_norm_out.weight'] = vae_state_dict[
        'decoder.norm_out.weight']
    new_checkpoint['decoder.conv_norm_out.bias'] = vae_state_dict[
        'decoder.norm_out.bias']
    new_checkpoint['quant_conv.weight'] = vae_state_dict['quant_conv.weight']
    new_checkpoint['quant_conv.bias'] = vae_state_dict['quant_conv.bias']
    new_checkpoint['post_quant_conv.weight'] = vae_state_dict[
        'post_quant_conv.weight']
    new_checkpoint['post_quant_conv.bias'] = vae_state_dict[
        'post_quant_conv.bias']
    num_down_blocks = len({'.'.join(layer.split('.')[:3]) for layer in
        vae_state_dict if 'encoder.down' in layer})
    down_blocks = {layer_id: [key for key in vae_state_dict if 
        f'down.{layer_id}' in key] for layer_id in range(num_down_blocks)}
    num_up_blocks = len({'.'.join(layer.split('.')[:3]) for layer in
        vae_state_dict if 'decoder.up' in layer})
    up_blocks = {layer_id: [key for key in vae_state_dict if 
        f'up.{layer_id}' in key] for layer_id in range(num_up_blocks)}
    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f'down.{i}' in key and 
            f'down.{i}.downsample' not in key]
        if f'encoder.down.{i}.downsample.conv.weight' in vae_state_dict:
            new_checkpoint[
                f'encoder.down_blocks.{i}.downsamplers.0.conv.weight'
                ] = vae_state_dict.pop(
                f'encoder.down.{i}.downsample.conv.weight')
            new_checkpoint[f'encoder.down_blocks.{i}.downsamplers.0.conv.bias'
                ] = vae_state_dict.pop(f'encoder.down.{i}.downsample.conv.bias'
                )
        paths = renew_vae_resnet_paths(resnets)
        meta_path = {'old': f'down.{i}.block', 'new':
            f'down_blocks.{i}.resnets'}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict,
            additional_replacements=[meta_path], config=config)
    mid_resnets = [key for key in vae_state_dict if 'encoder.mid.block' in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f'encoder.mid.block_{i}' in
            key]
        paths = renew_vae_resnet_paths(resnets)
        meta_path = {'old': f'mid.block_{i}', 'new':
            f'mid_block.resnets.{i - 1}'}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict,
            additional_replacements=[meta_path], config=config)
    mid_attentions = [key for key in vae_state_dict if 'encoder.mid.attn' in
        key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {'old': 'mid.attn_1', 'new': 'mid_block.attentions.0'}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict,
        additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)
    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [key for key in up_blocks[block_id] if f'up.{block_id}' in
            key and f'up.{block_id}.upsample' not in key]
        if f'decoder.up.{block_id}.upsample.conv.weight' in vae_state_dict:
            new_checkpoint[f'decoder.up_blocks.{i}.upsamplers.0.conv.weight'
                ] = vae_state_dict[
                f'decoder.up.{block_id}.upsample.conv.weight']
            new_checkpoint[f'decoder.up_blocks.{i}.upsamplers.0.conv.bias'
                ] = vae_state_dict[f'decoder.up.{block_id}.upsample.conv.bias']
        paths = renew_vae_resnet_paths(resnets)
        meta_path = {'old': f'up.{block_id}.block', 'new':
            f'up_blocks.{i}.resnets'}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict,
            additional_replacements=[meta_path], config=config)
    mid_resnets = [key for key in vae_state_dict if 'decoder.mid.block' in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f'decoder.mid.block_{i}' in
            key]
        paths = renew_vae_resnet_paths(resnets)
        meta_path = {'old': f'mid.block_{i}', 'new':
            f'mid_block.resnets.{i - 1}'}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict,
            additional_replacements=[meta_path], config=config)
    mid_attentions = [key for key in vae_state_dict if 'decoder.mid.attn' in
        key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {'old': 'mid.attn_1', 'new': 'mid_block.attentions.0'}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict,
        additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint
