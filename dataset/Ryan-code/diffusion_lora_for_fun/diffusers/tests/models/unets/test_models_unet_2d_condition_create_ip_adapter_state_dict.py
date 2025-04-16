def create_ip_adapter_state_dict(model):
    ip_cross_attn_state_dict = {}
    key_id = 1
    for name in model.attn_processors.keys():
        cross_attention_dim = None if name.endswith('attn1.processor'
            ) or 'motion_module' in name else model.config.cross_attention_dim
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
    image_projection = ImageProjection(cross_attention_dim=
        cross_attention_dim, image_embed_dim=cross_attention_dim,
        num_image_text_embeds=4)
    ip_image_projection_state_dict = {}
    sd = image_projection.state_dict()
    ip_image_projection_state_dict.update({'proj.weight': sd[
        'image_embeds.weight'], 'proj.bias': sd['image_embeds.bias'],
        'norm.weight': sd['norm.weight'], 'norm.bias': sd['norm.bias']})
    del sd
    ip_state_dict = {}
    ip_state_dict.update({'image_proj': ip_image_projection_state_dict,
        'ip_adapter': ip_cross_attn_state_dict})
    return ip_state_dict
