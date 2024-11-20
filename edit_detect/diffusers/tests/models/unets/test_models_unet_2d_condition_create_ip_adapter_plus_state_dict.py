def create_ip_adapter_plus_state_dict(model):
    ip_cross_attn_state_dict = {}
    key_id = 1
    for name in model.attn_processors.keys():
        cross_attention_dim = None if name.endswith('attn1.processor'
            ) else model.config.cross_attention_dim
        if name.startswith('mid_block'):
            hidden_size = model.config.block_out_channels[-1]
        elif name.startswith('up_blocks'):
            block_id = int(name[len('up_blocks.')])
            hidden_size = list(reversed(model.config.block_out_channels))[
                block_id]
        elif name.startswith('down_blocks'):
            block_id = int(name[len('down_blocks.')])
            hidden_size = model.config.block_out_channels[block_id]
        if cross_attention_dim is not None:
            sd = IPAdapterAttnProcessor(hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim, scale=1.0).state_dict(
                )
            ip_cross_attn_state_dict.update({f'{key_id}.to_k_ip.weight': sd
                ['to_k_ip.0.weight'], f'{key_id}.to_v_ip.weight': sd[
                'to_v_ip.0.weight']})
            key_id += 2
    cross_attention_dim = model.config['cross_attention_dim']
    image_projection = IPAdapterPlusImageProjection(embed_dims=
        cross_attention_dim, output_dims=cross_attention_dim, dim_head=32,
        heads=2, num_queries=4)
    ip_image_projection_state_dict = OrderedDict()
    for k, v in image_projection.state_dict().items():
        if '2.to' in k:
            k = k.replace('2.to', '0.to')
        elif '3.0.weight' in k:
            k = k.replace('3.0.weight', '1.0.weight')
        elif '3.0.bias' in k:
            k = k.replace('3.0.bias', '1.0.bias')
        elif '3.0.weight' in k:
            k = k.replace('3.0.weight', '1.0.weight')
        elif '3.1.net.0.proj.weight' in k:
            k = k.replace('3.1.net.0.proj.weight', '1.1.weight')
        elif '3.net.2.weight' in k:
            k = k.replace('3.net.2.weight', '1.3.weight')
        elif 'layers.0.0' in k:
            k = k.replace('layers.0.0', 'layers.0.0.norm1')
        elif 'layers.0.1' in k:
            k = k.replace('layers.0.1', 'layers.0.0.norm2')
        elif 'layers.1.0' in k:
            k = k.replace('layers.1.0', 'layers.1.0.norm1')
        elif 'layers.1.1' in k:
            k = k.replace('layers.1.1', 'layers.1.0.norm2')
        elif 'layers.2.0' in k:
            k = k.replace('layers.2.0', 'layers.2.0.norm1')
        elif 'layers.2.1' in k:
            k = k.replace('layers.2.1', 'layers.2.0.norm2')
        if 'norm_cross' in k:
            ip_image_projection_state_dict[k.replace('norm_cross', 'norm1')
                ] = v
        elif 'layer_norm' in k:
            ip_image_projection_state_dict[k.replace('layer_norm', 'norm2')
                ] = v
        elif 'to_k' in k:
            ip_image_projection_state_dict[k.replace('to_k', 'to_kv')
                ] = torch.cat([v, v], dim=0)
        elif 'to_v' in k:
            continue
        elif 'to_out.0' in k:
            ip_image_projection_state_dict[k.replace('to_out.0', 'to_out')] = v
        else:
            ip_image_projection_state_dict[k] = v
    ip_state_dict = {}
    ip_state_dict.update({'image_proj': ip_image_projection_state_dict,
        'ip_adapter': ip_cross_attn_state_dict})
    return ip_state_dict
