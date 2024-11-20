def convert_ldm_unet_checkpoint(checkpoint, config, extract_ema=False, **kwargs
    ):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """
    unet_state_dict = {}
    keys = list(checkpoint.keys())
    unet_key = LDM_UNET_KEY
    if sum(k.startswith('model_ema') for k in keys) > 100 and extract_ema:
        logger.warninging('Checkpoint has both EMA and non-EMA weights.')
        logger.warninging(
            'In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag.'
            )
        for key in keys:
            if key.startswith('model.diffusion_model'):
                flat_ema_key = 'model_ema.' + ''.join(key.split('.')[1:])
                unet_state_dict[key.replace(unet_key, '')] = checkpoint.get(
                    flat_ema_key)
    else:
        if sum(k.startswith('model_ema') for k in keys) > 100:
            logger.warninging(
                'In this conversion only the non-EMA weights are extracted. If you want to instead extract the EMA weights (usually better for inference), please make sure to add the `--extract_ema` flag.'
                )
        for key in keys:
            if key.startswith(unet_key):
                unet_state_dict[key.replace(unet_key, '')] = checkpoint.get(key
                    )
    new_checkpoint = {}
    ldm_unet_keys = DIFFUSERS_TO_LDM_MAPPING['unet']['layers']
    for diffusers_key, ldm_key in ldm_unet_keys.items():
        if ldm_key not in unet_state_dict:
            continue
        new_checkpoint[diffusers_key] = unet_state_dict[ldm_key]
    if 'class_embed_type' in config and config['class_embed_type'] in [
        'timestep', 'projection']:
        class_embed_keys = DIFFUSERS_TO_LDM_MAPPING['unet']['class_embed_type']
        for diffusers_key, ldm_key in class_embed_keys.items():
            new_checkpoint[diffusers_key] = unet_state_dict[ldm_key]
    if 'addition_embed_type' in config and config['addition_embed_type'
        ] == 'text_time':
        addition_embed_keys = DIFFUSERS_TO_LDM_MAPPING['unet'][
            'addition_embed_type']
        for diffusers_key, ldm_key in addition_embed_keys.items():
            new_checkpoint[diffusers_key] = unet_state_dict[ldm_key]
    if 'num_class_embeds' in config:
        if config['num_class_embeds'
            ] is not None and 'label_emb.weight' in unet_state_dict:
            new_checkpoint['class_embedding.weight'] = unet_state_dict[
                'label_emb.weight']
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
        update_unet_resnet_ldm_to_diffusers(resnets, new_checkpoint,
            unet_state_dict, {'old': f'input_blocks.{i}.0', 'new':
            f'down_blocks.{block_id}.resnets.{layer_in_block_id}'})
        if f'input_blocks.{i}.0.op.weight' in unet_state_dict:
            new_checkpoint[f'down_blocks.{block_id}.downsamplers.0.conv.weight'
                ] = unet_state_dict.get(f'input_blocks.{i}.0.op.weight')
            new_checkpoint[f'down_blocks.{block_id}.downsamplers.0.conv.bias'
                ] = unet_state_dict.get(f'input_blocks.{i}.0.op.bias')
        attentions = [key for key in input_blocks[i] if 
            f'input_blocks.{i}.1' in key]
        if attentions:
            update_unet_attention_ldm_to_diffusers(attentions,
                new_checkpoint, unet_state_dict, {'old':
                f'input_blocks.{i}.1', 'new':
                f'down_blocks.{block_id}.attentions.{layer_in_block_id}'})
    for key in middle_blocks.keys():
        diffusers_key = max(key - 1, 0)
        if key % 2 == 0:
            update_unet_resnet_ldm_to_diffusers(middle_blocks[key],
                new_checkpoint, unet_state_dict, mapping={'old':
                f'middle_block.{key}', 'new':
                f'mid_block.resnets.{diffusers_key}'})
        else:
            update_unet_attention_ldm_to_diffusers(middle_blocks[key],
                new_checkpoint, unet_state_dict, mapping={'old':
                f'middle_block.{key}', 'new':
                f'mid_block.attentions.{diffusers_key}'})
    for i in range(num_output_blocks):
        block_id = i // (config['layers_per_block'] + 1)
        layer_in_block_id = i % (config['layers_per_block'] + 1)
        resnets = [key for key in output_blocks[i] if 
            f'output_blocks.{i}.0' in key and f'output_blocks.{i}.0.op' not in
            key]
        update_unet_resnet_ldm_to_diffusers(resnets, new_checkpoint,
            unet_state_dict, {'old': f'output_blocks.{i}.0', 'new':
            f'up_blocks.{block_id}.resnets.{layer_in_block_id}'})
        attentions = [key for key in output_blocks[i] if 
            f'output_blocks.{i}.1' in key and f'output_blocks.{i}.1.conv'
             not in key]
        if attentions:
            update_unet_attention_ldm_to_diffusers(attentions,
                new_checkpoint, unet_state_dict, {'old':
                f'output_blocks.{i}.1', 'new':
                f'up_blocks.{block_id}.attentions.{layer_in_block_id}'})
        if f'output_blocks.{i}.1.conv.weight' in unet_state_dict:
            new_checkpoint[f'up_blocks.{block_id}.upsamplers.0.conv.weight'
                ] = unet_state_dict[f'output_blocks.{i}.1.conv.weight']
            new_checkpoint[f'up_blocks.{block_id}.upsamplers.0.conv.bias'
                ] = unet_state_dict[f'output_blocks.{i}.1.conv.bias']
        if f'output_blocks.{i}.2.conv.weight' in unet_state_dict:
            new_checkpoint[f'up_blocks.{block_id}.upsamplers.0.conv.weight'
                ] = unet_state_dict[f'output_blocks.{i}.2.conv.weight']
            new_checkpoint[f'up_blocks.{block_id}.upsamplers.0.conv.bias'
                ] = unet_state_dict[f'output_blocks.{i}.2.conv.bias']
    return new_checkpoint
