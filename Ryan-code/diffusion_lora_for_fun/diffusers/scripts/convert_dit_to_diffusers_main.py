def main(args):
    state_dict = download_model(pretrained_models[args.image_size])
    state_dict['pos_embed.proj.weight'] = state_dict['x_embedder.proj.weight']
    state_dict['pos_embed.proj.bias'] = state_dict['x_embedder.proj.bias']
    state_dict.pop('x_embedder.proj.weight')
    state_dict.pop('x_embedder.proj.bias')
    for depth in range(28):
        state_dict[
            f'transformer_blocks.{depth}.norm1.emb.timestep_embedder.linear_1.weight'
            ] = state_dict['t_embedder.mlp.0.weight']
        state_dict[
            f'transformer_blocks.{depth}.norm1.emb.timestep_embedder.linear_1.bias'
            ] = state_dict['t_embedder.mlp.0.bias']
        state_dict[
            f'transformer_blocks.{depth}.norm1.emb.timestep_embedder.linear_2.weight'
            ] = state_dict['t_embedder.mlp.2.weight']
        state_dict[
            f'transformer_blocks.{depth}.norm1.emb.timestep_embedder.linear_2.bias'
            ] = state_dict['t_embedder.mlp.2.bias']
        state_dict[
            f'transformer_blocks.{depth}.norm1.emb.class_embedder.embedding_table.weight'
            ] = state_dict['y_embedder.embedding_table.weight']
        state_dict[f'transformer_blocks.{depth}.norm1.linear.weight'
            ] = state_dict[f'blocks.{depth}.adaLN_modulation.1.weight']
        state_dict[f'transformer_blocks.{depth}.norm1.linear.bias'
            ] = state_dict[f'blocks.{depth}.adaLN_modulation.1.bias']
        q, k, v = torch.chunk(state_dict[f'blocks.{depth}.attn.qkv.weight'],
            3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(state_dict[
            f'blocks.{depth}.attn.qkv.bias'], 3, dim=0)
        state_dict[f'transformer_blocks.{depth}.attn1.to_q.weight'] = q
        state_dict[f'transformer_blocks.{depth}.attn1.to_q.bias'] = q_bias
        state_dict[f'transformer_blocks.{depth}.attn1.to_k.weight'] = k
        state_dict[f'transformer_blocks.{depth}.attn1.to_k.bias'] = k_bias
        state_dict[f'transformer_blocks.{depth}.attn1.to_v.weight'] = v
        state_dict[f'transformer_blocks.{depth}.attn1.to_v.bias'] = v_bias
        state_dict[f'transformer_blocks.{depth}.attn1.to_out.0.weight'
            ] = state_dict[f'blocks.{depth}.attn.proj.weight']
        state_dict[f'transformer_blocks.{depth}.attn1.to_out.0.bias'
            ] = state_dict[f'blocks.{depth}.attn.proj.bias']
        state_dict[f'transformer_blocks.{depth}.ff.net.0.proj.weight'
            ] = state_dict[f'blocks.{depth}.mlp.fc1.weight']
        state_dict[f'transformer_blocks.{depth}.ff.net.0.proj.bias'
            ] = state_dict[f'blocks.{depth}.mlp.fc1.bias']
        state_dict[f'transformer_blocks.{depth}.ff.net.2.weight'] = state_dict[
            f'blocks.{depth}.mlp.fc2.weight']
        state_dict[f'transformer_blocks.{depth}.ff.net.2.bias'] = state_dict[
            f'blocks.{depth}.mlp.fc2.bias']
        state_dict.pop(f'blocks.{depth}.attn.qkv.weight')
        state_dict.pop(f'blocks.{depth}.attn.qkv.bias')
        state_dict.pop(f'blocks.{depth}.attn.proj.weight')
        state_dict.pop(f'blocks.{depth}.attn.proj.bias')
        state_dict.pop(f'blocks.{depth}.mlp.fc1.weight')
        state_dict.pop(f'blocks.{depth}.mlp.fc1.bias')
        state_dict.pop(f'blocks.{depth}.mlp.fc2.weight')
        state_dict.pop(f'blocks.{depth}.mlp.fc2.bias')
        state_dict.pop(f'blocks.{depth}.adaLN_modulation.1.weight')
        state_dict.pop(f'blocks.{depth}.adaLN_modulation.1.bias')
    state_dict.pop('t_embedder.mlp.0.weight')
    state_dict.pop('t_embedder.mlp.0.bias')
    state_dict.pop('t_embedder.mlp.2.weight')
    state_dict.pop('t_embedder.mlp.2.bias')
    state_dict.pop('y_embedder.embedding_table.weight')
    state_dict['proj_out_1.weight'] = state_dict[
        'final_layer.adaLN_modulation.1.weight']
    state_dict['proj_out_1.bias'] = state_dict[
        'final_layer.adaLN_modulation.1.bias']
    state_dict['proj_out_2.weight'] = state_dict['final_layer.linear.weight']
    state_dict['proj_out_2.bias'] = state_dict['final_layer.linear.bias']
    state_dict.pop('final_layer.linear.weight')
    state_dict.pop('final_layer.linear.bias')
    state_dict.pop('final_layer.adaLN_modulation.1.weight')
    state_dict.pop('final_layer.adaLN_modulation.1.bias')
    transformer = Transformer2DModel(sample_size=args.image_size // 8,
        num_layers=28, attention_head_dim=72, in_channels=4, out_channels=8,
        patch_size=2, attention_bias=True, num_attention_heads=16,
        activation_fn='gelu-approximate', num_embeds_ada_norm=1000,
        norm_type='ada_norm_zero', norm_elementwise_affine=False)
    transformer.load_state_dict(state_dict, strict=True)
    scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule=
        'linear', prediction_type='epsilon', clip_sample=False)
    vae = AutoencoderKL.from_pretrained(args.vae_model)
    pipeline = DiTPipeline(transformer=transformer, vae=vae, scheduler=
        scheduler)
    if args.save:
        pipeline.save_pretrained(args.checkpoint_path)
