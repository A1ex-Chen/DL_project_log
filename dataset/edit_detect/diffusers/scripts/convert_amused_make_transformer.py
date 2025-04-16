def make_transformer(old_transformer, model_256):
    args = dict(old_transformer.config)
    force_down_up_sample = args['force_down_up_sample']
    signature = inspect.signature(UVit2DModel.__init__)
    args_ = {'downsample': force_down_up_sample, 'upsample':
        force_down_up_sample, 'block_out_channels': args[
        'block_out_channels'][0], 'sample_size': 16 if model_256 else 32}
    for s in list(signature.parameters.keys()):
        if s in ['self', 'downsample', 'upsample', 'sample_size',
            'block_out_channels']:
            continue
        args_[s] = args[s]
    new_transformer = UVit2DModel(**args_)
    new_transformer.to(device)
    new_transformer.set_attn_processor(AttnProcessor())
    state_dict = old_transformer.state_dict()
    state_dict['cond_embed.linear_1.weight'] = state_dict.pop(
        'cond_embed.0.weight')
    state_dict['cond_embed.linear_2.weight'] = state_dict.pop(
        'cond_embed.2.weight')
    for i in range(22):
        state_dict[f'transformer_layers.{i}.norm1.norm.weight'
            ] = state_dict.pop(f'transformer_layers.{i}.attn_layer_norm.weight'
            )
        state_dict[f'transformer_layers.{i}.norm1.linear.weight'
            ] = state_dict.pop(
            f'transformer_layers.{i}.self_attn_adaLN_modulation.mapper.weight')
        state_dict[f'transformer_layers.{i}.attn1.to_q.weight'
            ] = state_dict.pop(f'transformer_layers.{i}.attention.query.weight'
            )
        state_dict[f'transformer_layers.{i}.attn1.to_k.weight'
            ] = state_dict.pop(f'transformer_layers.{i}.attention.key.weight')
        state_dict[f'transformer_layers.{i}.attn1.to_v.weight'
            ] = state_dict.pop(f'transformer_layers.{i}.attention.value.weight'
            )
        state_dict[f'transformer_layers.{i}.attn1.to_out.0.weight'
            ] = state_dict.pop(f'transformer_layers.{i}.attention.out.weight')
        state_dict[f'transformer_layers.{i}.norm2.norm.weight'
            ] = state_dict.pop(
            f'transformer_layers.{i}.crossattn_layer_norm.weight')
        state_dict[f'transformer_layers.{i}.norm2.linear.weight'
            ] = state_dict.pop(
            f'transformer_layers.{i}.cross_attn_adaLN_modulation.mapper.weight'
            )
        state_dict[f'transformer_layers.{i}.attn2.to_q.weight'
            ] = state_dict.pop(
            f'transformer_layers.{i}.crossattention.query.weight')
        state_dict[f'transformer_layers.{i}.attn2.to_k.weight'
            ] = state_dict.pop(
            f'transformer_layers.{i}.crossattention.key.weight')
        state_dict[f'transformer_layers.{i}.attn2.to_v.weight'
            ] = state_dict.pop(
            f'transformer_layers.{i}.crossattention.value.weight')
        state_dict[f'transformer_layers.{i}.attn2.to_out.0.weight'
            ] = state_dict.pop(
            f'transformer_layers.{i}.crossattention.out.weight')
        state_dict[f'transformer_layers.{i}.norm3.norm.weight'
            ] = state_dict.pop(
            f'transformer_layers.{i}.ffn.pre_mlp_layer_norm.weight')
        state_dict[f'transformer_layers.{i}.norm3.linear.weight'
            ] = state_dict.pop(
            f'transformer_layers.{i}.ffn.adaLN_modulation.mapper.weight')
        wi_0_weight = state_dict.pop(f'transformer_layers.{i}.ffn.wi_0.weight')
        wi_1_weight = state_dict.pop(f'transformer_layers.{i}.ffn.wi_1.weight')
        proj_weight = torch.concat([wi_1_weight, wi_0_weight], dim=0)
        state_dict[f'transformer_layers.{i}.ff.net.0.proj.weight'
            ] = proj_weight
        state_dict[f'transformer_layers.{i}.ff.net.2.weight'] = state_dict.pop(
            f'transformer_layers.{i}.ffn.wo.weight')
    if force_down_up_sample:
        state_dict['down_block.downsample.norm.weight'] = state_dict.pop(
            'down_blocks.0.downsample.0.norm.weight')
        state_dict['down_block.downsample.conv.weight'] = state_dict.pop(
            'down_blocks.0.downsample.1.weight')
        state_dict['up_block.upsample.norm.weight'] = state_dict.pop(
            'up_blocks.0.upsample.0.norm.weight')
        state_dict['up_block.upsample.conv.weight'] = state_dict.pop(
            'up_blocks.0.upsample.1.weight')
    state_dict['mlm_layer.layer_norm.weight'] = state_dict.pop(
        'mlm_layer.layer_norm.norm.weight')
    for i in range(3):
        state_dict[f'down_block.res_blocks.{i}.norm.weight'] = state_dict.pop(
            f'down_blocks.0.res_blocks.{i}.norm.norm.weight')
        state_dict[f'down_block.res_blocks.{i}.channelwise_linear_1.weight'
            ] = state_dict.pop(
            f'down_blocks.0.res_blocks.{i}.channelwise.0.weight')
        state_dict[f'down_block.res_blocks.{i}.channelwise_norm.gamma'
            ] = state_dict.pop(
            f'down_blocks.0.res_blocks.{i}.channelwise.2.gamma')
        state_dict[f'down_block.res_blocks.{i}.channelwise_norm.beta'
            ] = state_dict.pop(
            f'down_blocks.0.res_blocks.{i}.channelwise.2.beta')
        state_dict[f'down_block.res_blocks.{i}.channelwise_linear_2.weight'
            ] = state_dict.pop(
            f'down_blocks.0.res_blocks.{i}.channelwise.4.weight')
        state_dict[f'down_block.res_blocks.{i}.cond_embeds_mapper.weight'
            ] = state_dict.pop(
            f'down_blocks.0.res_blocks.{i}.adaLN_modulation.mapper.weight')
        state_dict[f'down_block.attention_blocks.{i}.norm1.weight'
            ] = state_dict.pop(
            f'down_blocks.0.attention_blocks.{i}.attn_layer_norm.weight')
        state_dict[f'down_block.attention_blocks.{i}.attn1.to_q.weight'
            ] = state_dict.pop(
            f'down_blocks.0.attention_blocks.{i}.attention.query.weight')
        state_dict[f'down_block.attention_blocks.{i}.attn1.to_k.weight'
            ] = state_dict.pop(
            f'down_blocks.0.attention_blocks.{i}.attention.key.weight')
        state_dict[f'down_block.attention_blocks.{i}.attn1.to_v.weight'
            ] = state_dict.pop(
            f'down_blocks.0.attention_blocks.{i}.attention.value.weight')
        state_dict[f'down_block.attention_blocks.{i}.attn1.to_out.0.weight'
            ] = state_dict.pop(
            f'down_blocks.0.attention_blocks.{i}.attention.out.weight')
        state_dict[f'down_block.attention_blocks.{i}.norm2.weight'
            ] = state_dict.pop(
            f'down_blocks.0.attention_blocks.{i}.crossattn_layer_norm.weight')
        state_dict[f'down_block.attention_blocks.{i}.attn2.to_q.weight'
            ] = state_dict.pop(
            f'down_blocks.0.attention_blocks.{i}.crossattention.query.weight')
        state_dict[f'down_block.attention_blocks.{i}.attn2.to_k.weight'
            ] = state_dict.pop(
            f'down_blocks.0.attention_blocks.{i}.crossattention.key.weight')
        state_dict[f'down_block.attention_blocks.{i}.attn2.to_v.weight'
            ] = state_dict.pop(
            f'down_blocks.0.attention_blocks.{i}.crossattention.value.weight')
        state_dict[f'down_block.attention_blocks.{i}.attn2.to_out.0.weight'
            ] = state_dict.pop(
            f'down_blocks.0.attention_blocks.{i}.crossattention.out.weight')
        state_dict[f'up_block.res_blocks.{i}.norm.weight'] = state_dict.pop(
            f'up_blocks.0.res_blocks.{i}.norm.norm.weight')
        state_dict[f'up_block.res_blocks.{i}.channelwise_linear_1.weight'
            ] = state_dict.pop(
            f'up_blocks.0.res_blocks.{i}.channelwise.0.weight')
        state_dict[f'up_block.res_blocks.{i}.channelwise_norm.gamma'
            ] = state_dict.pop(
            f'up_blocks.0.res_blocks.{i}.channelwise.2.gamma')
        state_dict[f'up_block.res_blocks.{i}.channelwise_norm.beta'
            ] = state_dict.pop(f'up_blocks.0.res_blocks.{i}.channelwise.2.beta'
            )
        state_dict[f'up_block.res_blocks.{i}.channelwise_linear_2.weight'
            ] = state_dict.pop(
            f'up_blocks.0.res_blocks.{i}.channelwise.4.weight')
        state_dict[f'up_block.res_blocks.{i}.cond_embeds_mapper.weight'
            ] = state_dict.pop(
            f'up_blocks.0.res_blocks.{i}.adaLN_modulation.mapper.weight')
        state_dict[f'up_block.attention_blocks.{i}.norm1.weight'
            ] = state_dict.pop(
            f'up_blocks.0.attention_blocks.{i}.attn_layer_norm.weight')
        state_dict[f'up_block.attention_blocks.{i}.attn1.to_q.weight'
            ] = state_dict.pop(
            f'up_blocks.0.attention_blocks.{i}.attention.query.weight')
        state_dict[f'up_block.attention_blocks.{i}.attn1.to_k.weight'
            ] = state_dict.pop(
            f'up_blocks.0.attention_blocks.{i}.attention.key.weight')
        state_dict[f'up_block.attention_blocks.{i}.attn1.to_v.weight'
            ] = state_dict.pop(
            f'up_blocks.0.attention_blocks.{i}.attention.value.weight')
        state_dict[f'up_block.attention_blocks.{i}.attn1.to_out.0.weight'
            ] = state_dict.pop(
            f'up_blocks.0.attention_blocks.{i}.attention.out.weight')
        state_dict[f'up_block.attention_blocks.{i}.norm2.weight'
            ] = state_dict.pop(
            f'up_blocks.0.attention_blocks.{i}.crossattn_layer_norm.weight')
        state_dict[f'up_block.attention_blocks.{i}.attn2.to_q.weight'
            ] = state_dict.pop(
            f'up_blocks.0.attention_blocks.{i}.crossattention.query.weight')
        state_dict[f'up_block.attention_blocks.{i}.attn2.to_k.weight'
            ] = state_dict.pop(
            f'up_blocks.0.attention_blocks.{i}.crossattention.key.weight')
        state_dict[f'up_block.attention_blocks.{i}.attn2.to_v.weight'
            ] = state_dict.pop(
            f'up_blocks.0.attention_blocks.{i}.crossattention.value.weight')
        state_dict[f'up_block.attention_blocks.{i}.attn2.to_out.0.weight'
            ] = state_dict.pop(
            f'up_blocks.0.attention_blocks.{i}.crossattention.out.weight')
    for key in list(state_dict.keys()):
        if key.startswith('up_blocks.0'):
            key_ = 'up_block.' + '.'.join(key.split('.')[2:])
            state_dict[key_] = state_dict.pop(key)
        if key.startswith('down_blocks.0'):
            key_ = 'down_block.' + '.'.join(key.split('.')[2:])
            state_dict[key_] = state_dict.pop(key)
    new_transformer.load_state_dict(state_dict)
    input_ids = torch.randint(0, 10, (1, 32, 32), device=old_transformer.device
        )
    encoder_hidden_states = torch.randn((1, 77, 768), device=
        old_transformer.device)
    cond_embeds = torch.randn((1, 768), device=old_transformer.device)
    micro_conds = torch.tensor([[512, 512, 0, 0, 6]], dtype=torch.float32,
        device=old_transformer.device)
    old_out = old_transformer(input_ids.reshape(1, -1),
        encoder_hidden_states, cond_embeds, micro_conds)
    old_out = old_out.reshape(1, 32, 32, 8192).permute(0, 3, 1, 2)
    new_out = new_transformer(input_ids, encoder_hidden_states, cond_embeds,
        micro_conds)
    max_diff = (old_out - new_out).abs().max()
    total_diff = (old_out - new_out).abs().sum()
    print(f'Transformer max_diff: {max_diff} total_diff:  {total_diff}')
    assert max_diff < 0.01
    assert total_diff < 1500
    return new_transformer
