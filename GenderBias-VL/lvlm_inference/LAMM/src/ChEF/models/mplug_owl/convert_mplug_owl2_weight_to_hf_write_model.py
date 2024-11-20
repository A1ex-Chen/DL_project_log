def write_model(model_path, input_base_path, model_size, num_input_shards=1,
    num_output_shards=2, skip_permute=True, norm_eps=1e-05):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = model_path
    os.makedirs(tmp_model_path, exist_ok=True)
    num_shards = num_input_shards
    n_layers = llama_s2layer[model_size]
    n_heads = llama_s2heads[model_size]
    n_heads_per_shard = n_heads // num_shards
    n_dense = llama_s2dense[model_size]
    n_hidden = llama_s2hidden[model_size]
    hidden_per_head = n_hidden // n_heads
    base = 10000.0
    inv_freq = 1.0 / base ** (torch.arange(0, hidden_per_head, 2).float() /
        hidden_per_head)

    def permute(w, skip_permute=skip_permute):
        if skip_permute:
            return w
        return w.view(n_heads, n_hidden // n_heads // 2, 2, n_hidden
            ).transpose(1, 2).reshape(n_hidden, n_hidden)
    print(f'Fetching all parameters from the checkpoint at {input_base_path}.')
    if num_shards == 1:
        if os.path.exists(os.path.join(input_base_path, 'release')):
            filename = os.path.join(input_base_path, 'release',
                'mp_rank_00', 'model_optim_rng.pt')
        elif input_base_path.split('/')[-1].startswith('iter_'):
            iteration = eval(input_base_path.split('/')[-1].replace('iter_',
                '').lstrip('0'))
            load_dir = '/'.join(input_base_path.split('/')[:-1])
            filename = os.path.join(input_base_path, 'mp_rank_00',
                'model_optim_rng.pt')
            if not os.path.exists(filename):
                filename = filename.replace('model_optim_rng.pt',
                    'model_rng.pt')
        else:
            tracker_filename = os.path.join(input_base_path,
                'latest_checkpointed_iteration.txt')
            with open(tracker_filename, 'r') as f:
                metastring = f.read().strip()
            iteration = 'iter_{:07d}'.format(int(metastring))
            filename = os.path.join(input_base_path, iteration,
                'mp_rank_00', 'model_optim_rng.pt')
        if not os.path.exists(filename):
            filename = filename.replace('model_optim_rng.pt', 'model_rng.pt')
        original_filename = filename
        loaded = torch.load(filename, map_location='cpu')['model'][
            'language_model']
    else:
        filenames = []
        for i in range(num_shards):
            if os.path.exists(os.path.join(input_base_path, 'release')):
                filename = os.path.join(input_base_path, 'release',
                    f'mp_rank_{i:02d}', 'model_optim_rng.pt')
            else:
                tracker_filename = os.path.join(input_base_path,
                    'latest_checkpointed_iteration.txt')
                with open(tracker_filename, 'r') as f:
                    metastring = f.read().strip()
                iteration = 'iter_{:07d}'.format(int(metastring))
                filename = os.path.join(input_base_path, iteration,
                    f'mp_rank_{i:02d}', 'model_optim_rng.pt')
            if not os.path.exists(filename):
                filename = filename.replace('model_optim_rng.pt',
                    'model_rng.pt')
            filenames.append(filename)
        loaded = [torch.load(filenames[i], map_location='cpu')['model'][
            'language_model'] for i in range(num_shards)]
    print('Llama-Megatron Loaded!')
    param_count = 0
    index_dict = {'weight_map': {}}
    print(f'Weighted Converting for {n_layers} layers...')
    for layer_i in range(n_layers):
        print(layer_i)
        filename = f'pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin'
        if num_shards == 1:
            state_dict = {f'model.layers.{layer_i}.self_attn.q_proj.weight':
                loaded['encoder'][
                f'layers.{layer_i}.self_attention.q_proj.weight'],
                f'model.layers.{layer_i}.self_attn.k_proj.multiway.0.weight':
                loaded['encoder'][
                f'layers.{layer_i}.self_attention.k_proj.multiway.0.weight'
                ],
                f'model.layers.{layer_i}.self_attn.v_proj.multiway.0.weight':
                loaded['encoder'][
                f'layers.{layer_i}.self_attention.v_proj.multiway.0.weight'
                ],
                f'model.layers.{layer_i}.self_attn.k_proj.multiway.1.weight':
                loaded['encoder'][
                f'layers.{layer_i}.self_attention.k_proj.multiway.1.weight'
                ],
                f'model.layers.{layer_i}.self_attn.v_proj.multiway.1.weight':
                loaded['encoder'][
                f'layers.{layer_i}.self_attention.v_proj.multiway.1.weight'
                ], f'model.layers.{layer_i}.self_attn.o_proj.weight':
                loaded['encoder'][
                f'layers.{layer_i}.self_attention.o_proj.weight'],
                f'model.layers.{layer_i}.mlp.gate_proj.weight': loaded[
                'encoder'][f'layers.{layer_i}.mlp.gate_proj.weight'],
                f'model.layers.{layer_i}.mlp.down_proj.weight': loaded[
                'encoder'][f'layers.{layer_i}.mlp.down_proj.weight'],
                f'model.layers.{layer_i}.mlp.up_proj.weight': loaded[
                'encoder'][f'layers.{layer_i}.mlp.up_proj.weight'],
                f'model.layers.{layer_i}.input_layernorm.multiway.0.weight':
                loaded['encoder'][
                f'layers.{layer_i}.input_layernorm.multiway.0.weight'],
                f'model.layers.{layer_i}.post_attention_layernorm.multiway.0.weight'
                : loaded['encoder'][
                f'layers.{layer_i}.post_attention_layernorm.multiway.0.weight'
                ],
                f'model.layers.{layer_i}.input_layernorm.multiway.1.weight':
                loaded['encoder'][
                f'layers.{layer_i}.input_layernorm.multiway.1.weight'],
                f'model.layers.{layer_i}.post_attention_layernorm.multiway.1.weight'
                : loaded['encoder'][
                f'layers.{layer_i}.post_attention_layernorm.multiway.1.weight']
                }
        else:
            raise NotImplemented
        state_dict[f'model.layers.{layer_i}.self_attn.rotary_emb.inv_freq'
            ] = inv_freq
        for k, v in state_dict.items():
            index_dict['weight_map'][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))
        print(f'Sharded file saved to {filename}')
    filename = f'pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin'
    if num_shards == 1:
        state_dict = {'model.embed_tokens.weight': loaded['embedding'][
            'word_embeddings']['weight'], 'model.norm.weight': loaded[
            'encoder']['norm.weight'], 'lm_head.weight': loaded['encoder'][
            'lm_head.weight']}
    else:
        state_dict = {'model.embed_tokens.weight': loaded[0]['embedding'][
            'word_embeddings']['weight'], 'model.norm.weight': loaded[0][
            'encoder']['norm.weight'], 'lm_head.weight': loaded[0][
            'encoder']['lm_head.weight']}
    loaded_all = torch.load(original_filename, map_location='cpu')['model']
    state_dict.update({'model.vision_model.embeddings.cls_token':
        loaded_all['vision_model']['cls_token'],
        'model.vision_model.embeddings.patch_embed.weight': loaded_all[
        'vision_model']['patch_embed']['weight'],
        'model.vision_model.embeddings.position_embedding': loaded_all[
        'vision_model']['position_embeddings'],
        'model.vision_model.embeddings.pre_layernorm.bias': loaded_all[
        'vision_model']['pre_layernorm']['bias'],
        'model.vision_model.embeddings.pre_layernorm.weight': loaded_all[
        'vision_model']['pre_layernorm']['weight'],
        'model.vision_model.post_layernorm.bias': loaded_all['vision_model'
        ]['transformer']['final_layernorm.bias'],
        'model.vision_model.post_layernorm.weight': loaded_all[
        'vision_model']['transformer']['final_layernorm.weight']})
    for v_layer_idx in range(24):
        state_dict.update({
            f'model.vision_model.encoder.layers.{v_layer_idx}.input_layernorm.bias'
            : loaded_all['vision_model']['transformer'][
            f'layers.{v_layer_idx}.input_layernorm.bias'],
            f'model.vision_model.encoder.layers.{v_layer_idx}.input_layernorm.weight'
            : loaded_all['vision_model']['transformer'][
            f'layers.{v_layer_idx}.input_layernorm.weight'],
            f'model.vision_model.encoder.layers.{v_layer_idx}.mlp.fc1.bias':
            loaded_all['vision_model']['transformer'][
            f'layers.{v_layer_idx}.mlp.dense_h_to_4h.bias'],
            f'model.vision_model.encoder.layers.{v_layer_idx}.mlp.fc1.weight':
            loaded_all['vision_model']['transformer'][
            f'layers.{v_layer_idx}.mlp.dense_h_to_4h.weight'],
            f'model.vision_model.encoder.layers.{v_layer_idx}.mlp.fc2.bias':
            loaded_all['vision_model']['transformer'][
            f'layers.{v_layer_idx}.mlp.dense_4h_to_h.bias'],
            f'model.vision_model.encoder.layers.{v_layer_idx}.mlp.fc2.weight':
            loaded_all['vision_model']['transformer'][
            f'layers.{v_layer_idx}.mlp.dense_4h_to_h.weight'],
            f'model.vision_model.encoder.layers.{v_layer_idx}.post_attention_layernorm.bias'
            : loaded_all['vision_model']['transformer'][
            f'layers.{v_layer_idx}.post_attention_layernorm.bias'],
            f'model.vision_model.encoder.layers.{v_layer_idx}.post_attention_layernorm.weight'
            : loaded_all['vision_model']['transformer'][
            f'layers.{v_layer_idx}.post_attention_layernorm.weight'],
            f'model.vision_model.encoder.layers.{v_layer_idx}.self_attn.dense.bias'
            : loaded_all['vision_model']['transformer'][
            f'layers.{v_layer_idx}.self_attention.dense.bias'],
            f'model.vision_model.encoder.layers.{v_layer_idx}.self_attn.dense.weight'
            : loaded_all['vision_model']['transformer'][
            f'layers.{v_layer_idx}.self_attention.dense.weight'],
            f'model.vision_model.encoder.layers.{v_layer_idx}.self_attn.query_key_value.bias'
            : loaded_all['vision_model']['transformer'][
            f'layers.{v_layer_idx}.self_attention.query_key_value.bias'],
            f'model.vision_model.encoder.layers.{v_layer_idx}.self_attn.query_key_value.weight'
            : loaded_all['vision_model']['transformer'][
            f'layers.{v_layer_idx}.self_attention.query_key_value.weight']})
    state_dict.update({'model.visual_abstractor.query_embeds': loaded_all[
        'vision_abstractor']['learnable_queries'],
        'model.visual_abstractor.visual_fc.bias': loaded_all[
        'vision_abstractor']['visual_fc']['bias'],
        'model.visual_abstractor.visual_fc.weight': loaded_all[
        'vision_abstractor']['visual_fc']['weight'],
        'model.visual_abstractor.vit_eos': loaded_all['vision_abstractor'][
        'vit_eos']})
    for v_layer_idx in range(6):
        state_dict.update({
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.attention.key.bias'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.self_attention.k_proj.bias'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.attention.key.weight'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.self_attention.k_proj.weight'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.attention.query.bias'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.self_attention.q_proj.bias'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.attention.query.weight'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.self_attention.q_proj.weight'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.attention.value.bias'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.self_attention.v_proj.bias'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.attention.value.weight'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.self_attention.v_proj.weight'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.norm1.bias'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.norm1.bias'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.norm1.weight'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.norm1.weight'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.normk.bias'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.normk.bias'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.normk.weight'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.normk.weight'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.mlp.ffn_ln.bias'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.mlp.ffn_ln.bias'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.mlp.ffn_ln.weight'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.mlp.ffn_ln.weight'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.mlp.w1.bias'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.mlp.w1.bias'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.mlp.w1.weight'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.mlp.w1.weight'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.mlp.w2.bias'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.mlp.w2.bias'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.mlp.w2.weight'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.mlp.w2.weight'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.mlp.w3.bias'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.mlp.w3.bias'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.mlp.w3.weight'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.mlp.w3.weight'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.norm2.bias'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.norm2.bias'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.norm2.weight'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.norm2.weight'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.out_proj.bias'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.self_attention.o_proj.bias'],
            f'model.visual_abstractor.encoder.layers.{v_layer_idx}.crossattention.output.out_proj.weight'
            : loaded_all['vision_abstractor']['transformer'][
            f'layers.{v_layer_idx}.self_attention.o_proj.weight']})
    for k, v in state_dict.items():
        index_dict['weight_map'][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))
    index_dict['metadata'] = {'total_size': param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path,
        'pytorch_model.bin.index.json'))
    config = MPLUGOwl2Config()
    config.save_pretrained(tmp_model_path)
    del state_dict
    del loaded
    del loaded_all
    gc.collect()
