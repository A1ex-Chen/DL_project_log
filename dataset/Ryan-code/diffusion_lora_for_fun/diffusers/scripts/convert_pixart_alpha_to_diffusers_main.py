def main(args):
    all_state_dict = torch.load(args.orig_ckpt_path, map_location='cpu')
    state_dict = all_state_dict.pop('state_dict')
    converted_state_dict = {}
    converted_state_dict['pos_embed.proj.weight'] = state_dict.pop(
        'x_embedder.proj.weight')
    converted_state_dict['pos_embed.proj.bias'] = state_dict.pop(
        'x_embedder.proj.bias')
    converted_state_dict['caption_projection.linear_1.weight'
        ] = state_dict.pop('y_embedder.y_proj.fc1.weight')
    converted_state_dict['caption_projection.linear_1.bias'] = state_dict.pop(
        'y_embedder.y_proj.fc1.bias')
    converted_state_dict['caption_projection.linear_2.weight'
        ] = state_dict.pop('y_embedder.y_proj.fc2.weight')
    converted_state_dict['caption_projection.linear_2.bias'] = state_dict.pop(
        'y_embedder.y_proj.fc2.bias')
    converted_state_dict['adaln_single.emb.timestep_embedder.linear_1.weight'
        ] = state_dict.pop('t_embedder.mlp.0.weight')
    converted_state_dict['adaln_single.emb.timestep_embedder.linear_1.bias'
        ] = state_dict.pop('t_embedder.mlp.0.bias')
    converted_state_dict['adaln_single.emb.timestep_embedder.linear_2.weight'
        ] = state_dict.pop('t_embedder.mlp.2.weight')
    converted_state_dict['adaln_single.emb.timestep_embedder.linear_2.bias'
        ] = state_dict.pop('t_embedder.mlp.2.bias')
    if args.image_size == 1024:
        converted_state_dict[
            'adaln_single.emb.resolution_embedder.linear_1.weight'
            ] = state_dict.pop('csize_embedder.mlp.0.weight')
        converted_state_dict[
            'adaln_single.emb.resolution_embedder.linear_1.bias'
            ] = state_dict.pop('csize_embedder.mlp.0.bias')
        converted_state_dict[
            'adaln_single.emb.resolution_embedder.linear_2.weight'
            ] = state_dict.pop('csize_embedder.mlp.2.weight')
        converted_state_dict[
            'adaln_single.emb.resolution_embedder.linear_2.bias'
            ] = state_dict.pop('csize_embedder.mlp.2.bias')
        converted_state_dict[
            'adaln_single.emb.aspect_ratio_embedder.linear_1.weight'
            ] = state_dict.pop('ar_embedder.mlp.0.weight')
        converted_state_dict[
            'adaln_single.emb.aspect_ratio_embedder.linear_1.bias'
            ] = state_dict.pop('ar_embedder.mlp.0.bias')
        converted_state_dict[
            'adaln_single.emb.aspect_ratio_embedder.linear_2.weight'
            ] = state_dict.pop('ar_embedder.mlp.2.weight')
        converted_state_dict[
            'adaln_single.emb.aspect_ratio_embedder.linear_2.bias'
            ] = state_dict.pop('ar_embedder.mlp.2.bias')
    converted_state_dict['adaln_single.linear.weight'] = state_dict.pop(
        't_block.1.weight')
    converted_state_dict['adaln_single.linear.bias'] = state_dict.pop(
        't_block.1.bias')
    for depth in range(28):
        converted_state_dict[f'transformer_blocks.{depth}.scale_shift_table'
            ] = state_dict.pop(f'blocks.{depth}.scale_shift_table')
        q, k, v = torch.chunk(state_dict.pop(
            f'blocks.{depth}.attn.qkv.weight'), 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(state_dict.pop(
            f'blocks.{depth}.attn.qkv.bias'), 3, dim=0)
        converted_state_dict[f'transformer_blocks.{depth}.attn1.to_q.weight'
            ] = q
        converted_state_dict[f'transformer_blocks.{depth}.attn1.to_q.bias'
            ] = q_bias
        converted_state_dict[f'transformer_blocks.{depth}.attn1.to_k.weight'
            ] = k
        converted_state_dict[f'transformer_blocks.{depth}.attn1.to_k.bias'
            ] = k_bias
        converted_state_dict[f'transformer_blocks.{depth}.attn1.to_v.weight'
            ] = v
        converted_state_dict[f'transformer_blocks.{depth}.attn1.to_v.bias'
            ] = v_bias
        converted_state_dict[
            f'transformer_blocks.{depth}.attn1.to_out.0.weight'
            ] = state_dict.pop(f'blocks.{depth}.attn.proj.weight')
        converted_state_dict[f'transformer_blocks.{depth}.attn1.to_out.0.bias'
            ] = state_dict.pop(f'blocks.{depth}.attn.proj.bias')
        converted_state_dict[f'transformer_blocks.{depth}.ff.net.0.proj.weight'
            ] = state_dict.pop(f'blocks.{depth}.mlp.fc1.weight')
        converted_state_dict[f'transformer_blocks.{depth}.ff.net.0.proj.bias'
            ] = state_dict.pop(f'blocks.{depth}.mlp.fc1.bias')
        converted_state_dict[f'transformer_blocks.{depth}.ff.net.2.weight'
            ] = state_dict.pop(f'blocks.{depth}.mlp.fc2.weight')
        converted_state_dict[f'transformer_blocks.{depth}.ff.net.2.bias'
            ] = state_dict.pop(f'blocks.{depth}.mlp.fc2.bias')
        q = state_dict.pop(f'blocks.{depth}.cross_attn.q_linear.weight')
        q_bias = state_dict.pop(f'blocks.{depth}.cross_attn.q_linear.bias')
        k, v = torch.chunk(state_dict.pop(
            f'blocks.{depth}.cross_attn.kv_linear.weight'), 2, dim=0)
        k_bias, v_bias = torch.chunk(state_dict.pop(
            f'blocks.{depth}.cross_attn.kv_linear.bias'), 2, dim=0)
        converted_state_dict[f'transformer_blocks.{depth}.attn2.to_q.weight'
            ] = q
        converted_state_dict[f'transformer_blocks.{depth}.attn2.to_q.bias'
            ] = q_bias
        converted_state_dict[f'transformer_blocks.{depth}.attn2.to_k.weight'
            ] = k
        converted_state_dict[f'transformer_blocks.{depth}.attn2.to_k.bias'
            ] = k_bias
        converted_state_dict[f'transformer_blocks.{depth}.attn2.to_v.weight'
            ] = v
        converted_state_dict[f'transformer_blocks.{depth}.attn2.to_v.bias'
            ] = v_bias
        converted_state_dict[
            f'transformer_blocks.{depth}.attn2.to_out.0.weight'
            ] = state_dict.pop(f'blocks.{depth}.cross_attn.proj.weight')
        converted_state_dict[f'transformer_blocks.{depth}.attn2.to_out.0.bias'
            ] = state_dict.pop(f'blocks.{depth}.cross_attn.proj.bias')
    converted_state_dict['proj_out.weight'] = state_dict.pop(
        'final_layer.linear.weight')
    converted_state_dict['proj_out.bias'] = state_dict.pop(
        'final_layer.linear.bias')
    converted_state_dict['scale_shift_table'] = state_dict.pop(
        'final_layer.scale_shift_table')
    transformer = Transformer2DModel(sample_size=args.image_size // 8,
        num_layers=28, attention_head_dim=72, in_channels=4, out_channels=8,
        patch_size=2, attention_bias=True, num_attention_heads=16,
        cross_attention_dim=1152, activation_fn='gelu-approximate',
        num_embeds_ada_norm=1000, norm_type='ada_norm_single',
        norm_elementwise_affine=False, norm_eps=1e-06, caption_channels=4096)
    transformer.load_state_dict(converted_state_dict, strict=True)
    assert transformer.pos_embed.pos_embed is not None
    state_dict.pop('pos_embed')
    state_dict.pop('y_embedder.y_embedding')
    assert len(state_dict
        ) == 0, f'State dict is not empty, {state_dict.keys()}'
    num_model_params = sum(p.numel() for p in transformer.parameters())
    print(f'Total number of transformer parameters: {num_model_params}')
    if args.only_transformer:
        transformer.save_pretrained(os.path.join(args.dump_path, 'transformer')
            )
    else:
        scheduler = DPMSolverMultistepScheduler()
        vae = AutoencoderKL.from_pretrained(ckpt_id, subfolder='sd-vae-ft-ema')
        tokenizer = T5Tokenizer.from_pretrained(ckpt_id, subfolder=
            't5-v1_1-xxl')
        text_encoder = T5EncoderModel.from_pretrained(ckpt_id, subfolder=
            't5-v1_1-xxl')
        pipeline = PixArtAlphaPipeline(tokenizer=tokenizer, text_encoder=
            text_encoder, transformer=transformer, vae=vae, scheduler=scheduler
            )
        pipeline.save_pretrained(args.dump_path)
