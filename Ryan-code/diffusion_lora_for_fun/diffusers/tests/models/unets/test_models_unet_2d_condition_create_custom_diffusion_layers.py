def create_custom_diffusion_layers(model, mock_weights: bool=True):
    train_kv = True
    train_q_out = True
    custom_diffusion_attn_procs = {}
    st = model.state_dict()
    for name, _ in model.attn_processors.items():
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
        layer_name = name.split('.processor')[0]
        weights = {'to_k_custom_diffusion.weight': st[layer_name +
            '.to_k.weight'], 'to_v_custom_diffusion.weight': st[layer_name +
            '.to_v.weight']}
        if train_q_out:
            weights['to_q_custom_diffusion.weight'] = st[layer_name +
                '.to_q.weight']
            weights['to_out_custom_diffusion.0.weight'] = st[layer_name +
                '.to_out.0.weight']
            weights['to_out_custom_diffusion.0.bias'] = st[layer_name +
                '.to_out.0.bias']
        if cross_attention_dim is not None:
            custom_diffusion_attn_procs[name] = CustomDiffusionAttnProcessor(
                train_kv=train_kv, train_q_out=train_q_out, hidden_size=
                hidden_size, cross_attention_dim=cross_attention_dim).to(model
                .device)
            custom_diffusion_attn_procs[name].load_state_dict(weights)
            if mock_weights:
                with torch.no_grad():
                    custom_diffusion_attn_procs[name
                        ].to_k_custom_diffusion.weight += 1
                    custom_diffusion_attn_procs[name
                        ].to_v_custom_diffusion.weight += 1
        else:
            custom_diffusion_attn_procs[name] = CustomDiffusionAttnProcessor(
                train_kv=False, train_q_out=False, hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim)
    del st
    return custom_diffusion_attn_procs
