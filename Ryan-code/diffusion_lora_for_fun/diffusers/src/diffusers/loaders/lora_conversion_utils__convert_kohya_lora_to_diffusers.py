def _convert_kohya_lora_to_diffusers(state_dict, unet_name='unet',
    text_encoder_name='text_encoder'):
    unet_state_dict = {}
    te_state_dict = {}
    te2_state_dict = {}
    network_alphas = {}
    is_unet_dora_lora = any('dora_scale' in k and 'lora_unet_' in k for k in
        state_dict)
    is_te_dora_lora = any('dora_scale' in k and ('lora_te_' in k or 
        'lora_te1_' in k) for k in state_dict)
    is_te2_dora_lora = any('dora_scale' in k and 'lora_te2_' in k for k in
        state_dict)
    if is_unet_dora_lora or is_te_dora_lora or is_te2_dora_lora:
        if is_peft_version('<', '0.9.0'):
            raise ValueError(
                'You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`.'
                )
    lora_keys = [k for k in state_dict.keys() if k.endswith('lora_down.weight')
        ]
    for key in lora_keys:
        lora_name = key.split('.')[0]
        lora_name_up = lora_name + '.lora_up.weight'
        lora_name_alpha = lora_name + '.alpha'
        if lora_name.startswith('lora_unet_'):
            diffusers_name = key.replace('lora_unet_', '').replace('_', '.')
            if 'input.blocks' in diffusers_name:
                diffusers_name = diffusers_name.replace('input.blocks',
                    'down_blocks')
            else:
                diffusers_name = diffusers_name.replace('down.blocks',
                    'down_blocks')
            if 'middle.block' in diffusers_name:
                diffusers_name = diffusers_name.replace('middle.block',
                    'mid_block')
            else:
                diffusers_name = diffusers_name.replace('mid.block',
                    'mid_block')
            if 'output.blocks' in diffusers_name:
                diffusers_name = diffusers_name.replace('output.blocks',
                    'up_blocks')
            else:
                diffusers_name = diffusers_name.replace('up.blocks',
                    'up_blocks')
            diffusers_name = diffusers_name.replace('transformer.blocks',
                'transformer_blocks')
            diffusers_name = diffusers_name.replace('to.q.lora', 'to_q_lora')
            diffusers_name = diffusers_name.replace('to.k.lora', 'to_k_lora')
            diffusers_name = diffusers_name.replace('to.v.lora', 'to_v_lora')
            diffusers_name = diffusers_name.replace('to.out.0.lora',
                'to_out_lora')
            diffusers_name = diffusers_name.replace('proj.in', 'proj_in')
            diffusers_name = diffusers_name.replace('proj.out', 'proj_out')
            diffusers_name = diffusers_name.replace('emb.layers',
                'time_emb_proj')
            if ('emb' in diffusers_name and 'time.emb.proj' not in
                diffusers_name):
                pattern = '\\.\\d+(?=\\D*$)'
                diffusers_name = re.sub(pattern, '', diffusers_name, count=1)
            if '.in.' in diffusers_name:
                diffusers_name = diffusers_name.replace('in.layers.2', 'conv1')
            if '.out.' in diffusers_name:
                diffusers_name = diffusers_name.replace('out.layers.3', 'conv2'
                    )
            if ('downsamplers' in diffusers_name or 'upsamplers' in
                diffusers_name):
                diffusers_name = diffusers_name.replace('op', 'conv')
            if 'skip' in diffusers_name:
                diffusers_name = diffusers_name.replace('skip.connection',
                    'conv_shortcut')
            if 'time.emb.proj' in diffusers_name:
                diffusers_name = diffusers_name.replace('time.emb.proj',
                    'time_emb_proj')
            if 'conv.shortcut' in diffusers_name:
                diffusers_name = diffusers_name.replace('conv.shortcut',
                    'conv_shortcut')
            if 'transformer_blocks' in diffusers_name:
                if 'attn1' in diffusers_name or 'attn2' in diffusers_name:
                    diffusers_name = diffusers_name.replace('attn1',
                        'attn1.processor')
                    diffusers_name = diffusers_name.replace('attn2',
                        'attn2.processor')
                    unet_state_dict[diffusers_name] = state_dict.pop(key)
                    unet_state_dict[diffusers_name.replace('.down.', '.up.')
                        ] = state_dict.pop(lora_name_up)
                elif 'ff' in diffusers_name:
                    unet_state_dict[diffusers_name] = state_dict.pop(key)
                    unet_state_dict[diffusers_name.replace('.down.', '.up.')
                        ] = state_dict.pop(lora_name_up)
            elif any(key in diffusers_name for key in ('proj_in', 'proj_out')):
                unet_state_dict[diffusers_name] = state_dict.pop(key)
                unet_state_dict[diffusers_name.replace('.down.', '.up.')
                    ] = state_dict.pop(lora_name_up)
            else:
                unet_state_dict[diffusers_name] = state_dict.pop(key)
                unet_state_dict[diffusers_name.replace('.down.', '.up.')
                    ] = state_dict.pop(lora_name_up)
            if is_unet_dora_lora:
                dora_scale_key_to_replace = ('_lora.down.' if '_lora.down.' in
                    diffusers_name else '.lora.down.')
                unet_state_dict[diffusers_name.replace(
                    dora_scale_key_to_replace, '.lora_magnitude_vector.')
                    ] = state_dict.pop(key.replace('lora_down.weight',
                    'dora_scale'))
        elif lora_name.startswith(('lora_te_', 'lora_te1_', 'lora_te2_')):
            if lora_name.startswith(('lora_te_', 'lora_te1_')):
                key_to_replace = 'lora_te_' if lora_name.startswith('lora_te_'
                    ) else 'lora_te1_'
            else:
                key_to_replace = 'lora_te2_'
            diffusers_name = key.replace(key_to_replace, '').replace('_', '.')
            diffusers_name = diffusers_name.replace('text.model', 'text_model')
            diffusers_name = diffusers_name.replace('self.attn', 'self_attn')
            diffusers_name = diffusers_name.replace('q.proj.lora', 'to_q_lora')
            diffusers_name = diffusers_name.replace('k.proj.lora', 'to_k_lora')
            diffusers_name = diffusers_name.replace('v.proj.lora', 'to_v_lora')
            diffusers_name = diffusers_name.replace('out.proj.lora',
                'to_out_lora')
            if 'self_attn' in diffusers_name:
                if lora_name.startswith(('lora_te_', 'lora_te1_')):
                    te_state_dict[diffusers_name] = state_dict.pop(key)
                    te_state_dict[diffusers_name.replace('.down.', '.up.')
                        ] = state_dict.pop(lora_name_up)
                else:
                    te2_state_dict[diffusers_name] = state_dict.pop(key)
                    te2_state_dict[diffusers_name.replace('.down.', '.up.')
                        ] = state_dict.pop(lora_name_up)
            elif 'mlp' in diffusers_name:
                diffusers_name = diffusers_name.replace('.lora.',
                    '.lora_linear_layer.')
                if lora_name.startswith(('lora_te_', 'lora_te1_')):
                    te_state_dict[diffusers_name] = state_dict.pop(key)
                    te_state_dict[diffusers_name.replace('.down.', '.up.')
                        ] = state_dict.pop(lora_name_up)
                else:
                    te2_state_dict[diffusers_name] = state_dict.pop(key)
                    te2_state_dict[diffusers_name.replace('.down.', '.up.')
                        ] = state_dict.pop(lora_name_up)
            if (is_te_dora_lora or is_te2_dora_lora) and lora_name.startswith((
                'lora_te_', 'lora_te1_', 'lora_te2_')):
                dora_scale_key_to_replace_te = ('_lora.down.' if 
                    '_lora.down.' in diffusers_name else '.lora_linear_layer.')
                if lora_name.startswith(('lora_te_', 'lora_te1_')):
                    te_state_dict[diffusers_name.replace(
                        dora_scale_key_to_replace_te,
                        '.lora_magnitude_vector.')] = state_dict.pop(key.
                        replace('lora_down.weight', 'dora_scale'))
                elif lora_name.startswith('lora_te2_'):
                    te2_state_dict[diffusers_name.replace(
                        dora_scale_key_to_replace_te,
                        '.lora_magnitude_vector.')] = state_dict.pop(key.
                        replace('lora_down.weight', 'dora_scale'))
        if lora_name_alpha in state_dict:
            alpha = state_dict.pop(lora_name_alpha).item()
            if lora_name_alpha.startswith('lora_unet_'):
                prefix = 'unet.'
            elif lora_name_alpha.startswith(('lora_te_', 'lora_te1_')):
                prefix = 'text_encoder.'
            else:
                prefix = 'text_encoder_2.'
            new_name = prefix + diffusers_name.split('.lora.')[0] + '.alpha'
            network_alphas.update({new_name: alpha})
    if len(state_dict) > 0:
        raise ValueError(
            f"""The following keys have not been correctly be renamed: 

 {', '.join(state_dict.keys())}"""
            )
    logger.info('Kohya-style checkpoint detected.')
    unet_state_dict = {f'{unet_name}.{module_name}': params for module_name,
        params in unet_state_dict.items()}
    te_state_dict = {f'{text_encoder_name}.{module_name}': params for 
        module_name, params in te_state_dict.items()}
    te2_state_dict = {f'text_encoder_2.{module_name}': params for 
        module_name, params in te2_state_dict.items()} if len(te2_state_dict
        ) > 0 else None
    if te2_state_dict is not None:
        te_state_dict.update(te2_state_dict)
    new_state_dict = {**unet_state_dict, **te_state_dict}
    return new_state_dict, network_alphas
