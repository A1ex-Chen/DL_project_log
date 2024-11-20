def convert_ldm_vae_checkpoint(checkpoint, config):
    vae_state_dict = {}
    keys = list(checkpoint.keys())
    vae_key = LDM_VAE_KEY if any(k.startswith(LDM_VAE_KEY) for k in keys
        ) else ''
    for key in keys:
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, '')] = checkpoint.get(key)
    new_checkpoint = {}
    vae_diffusers_ldm_map = DIFFUSERS_TO_LDM_MAPPING['vae']
    for diffusers_key, ldm_key in vae_diffusers_ldm_map.items():
        if ldm_key not in vae_state_dict:
            continue
        new_checkpoint[diffusers_key] = vae_state_dict[ldm_key]
    num_down_blocks = len(config['down_block_types'])
    down_blocks = {layer_id: [key for key in vae_state_dict if 
        f'down.{layer_id}' in key] for layer_id in range(num_down_blocks)}
    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f'down.{i}' in key and 
            f'down.{i}.downsample' not in key]
        update_vae_resnet_ldm_to_diffusers(resnets, new_checkpoint,
            vae_state_dict, mapping={'old': f'down.{i}.block', 'new':
            f'down_blocks.{i}.resnets'})
        if f'encoder.down.{i}.downsample.conv.weight' in vae_state_dict:
            new_checkpoint[
                f'encoder.down_blocks.{i}.downsamplers.0.conv.weight'
                ] = vae_state_dict.get(
                f'encoder.down.{i}.downsample.conv.weight')
            new_checkpoint[f'encoder.down_blocks.{i}.downsamplers.0.conv.bias'
                ] = vae_state_dict.get(f'encoder.down.{i}.downsample.conv.bias'
                )
    mid_resnets = [key for key in vae_state_dict if 'encoder.mid.block' in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f'encoder.mid.block_{i}' in
            key]
        update_vae_resnet_ldm_to_diffusers(resnets, new_checkpoint,
            vae_state_dict, mapping={'old': f'mid.block_{i}', 'new':
            f'mid_block.resnets.{i - 1}'})
    mid_attentions = [key for key in vae_state_dict if 'encoder.mid.attn' in
        key]
    update_vae_attentions_ldm_to_diffusers(mid_attentions, new_checkpoint,
        vae_state_dict, mapping={'old': 'mid.attn_1', 'new':
        'mid_block.attentions.0'})
    num_up_blocks = len(config['up_block_types'])
    up_blocks = {layer_id: [key for key in vae_state_dict if 
        f'up.{layer_id}' in key] for layer_id in range(num_up_blocks)}
    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [key for key in up_blocks[block_id] if f'up.{block_id}' in
            key and f'up.{block_id}.upsample' not in key]
        update_vae_resnet_ldm_to_diffusers(resnets, new_checkpoint,
            vae_state_dict, mapping={'old': f'up.{block_id}.block', 'new':
            f'up_blocks.{i}.resnets'})
        if f'decoder.up.{block_id}.upsample.conv.weight' in vae_state_dict:
            new_checkpoint[f'decoder.up_blocks.{i}.upsamplers.0.conv.weight'
                ] = vae_state_dict[
                f'decoder.up.{block_id}.upsample.conv.weight']
            new_checkpoint[f'decoder.up_blocks.{i}.upsamplers.0.conv.bias'
                ] = vae_state_dict[f'decoder.up.{block_id}.upsample.conv.bias']
    mid_resnets = [key for key in vae_state_dict if 'decoder.mid.block' in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f'decoder.mid.block_{i}' in
            key]
        update_vae_resnet_ldm_to_diffusers(resnets, new_checkpoint,
            vae_state_dict, mapping={'old': f'mid.block_{i}', 'new':
            f'mid_block.resnets.{i - 1}'})
    mid_attentions = [key for key in vae_state_dict if 'decoder.mid.attn' in
        key]
    update_vae_attentions_ldm_to_diffusers(mid_attentions, new_checkpoint,
        vae_state_dict, mapping={'old': 'mid.attn_1', 'new':
        'mid_block.attentions.0'})
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint
