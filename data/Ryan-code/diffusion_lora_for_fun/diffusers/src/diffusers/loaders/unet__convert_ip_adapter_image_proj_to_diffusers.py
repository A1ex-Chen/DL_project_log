def _convert_ip_adapter_image_proj_to_diffusers(self, state_dict,
    low_cpu_mem_usage=False):
    if low_cpu_mem_usage:
        if is_accelerate_available():
            from accelerate import init_empty_weights
        else:
            low_cpu_mem_usage = False
            logger.warning(
                """Cannot initialize model with low cpu memory usage because `accelerate` was not found in the environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install `accelerate` for faster and less memory-intense model loading. You can do so with: 
```
pip install accelerate
```
."""
                )
    if low_cpu_mem_usage is True and not is_torch_version('>=', '1.9.0'):
        raise NotImplementedError(
            'Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set `low_cpu_mem_usage=False`.'
            )
    updated_state_dict = {}
    image_projection = None
    init_context = init_empty_weights if low_cpu_mem_usage else nullcontext
    if 'proj.weight' in state_dict:
        num_image_text_embeds = 4
        clip_embeddings_dim = state_dict['proj.weight'].shape[-1]
        cross_attention_dim = state_dict['proj.weight'].shape[0] // 4
        with init_context():
            image_projection = ImageProjection(cross_attention_dim=
                cross_attention_dim, image_embed_dim=clip_embeddings_dim,
                num_image_text_embeds=num_image_text_embeds)
        for key, value in state_dict.items():
            diffusers_name = key.replace('proj', 'image_embeds')
            updated_state_dict[diffusers_name] = value
    elif 'proj.3.weight' in state_dict:
        clip_embeddings_dim = state_dict['proj.0.weight'].shape[0]
        cross_attention_dim = state_dict['proj.3.weight'].shape[0]
        with init_context():
            image_projection = IPAdapterFullImageProjection(cross_attention_dim
                =cross_attention_dim, image_embed_dim=clip_embeddings_dim)
        for key, value in state_dict.items():
            diffusers_name = key.replace('proj.0', 'ff.net.0.proj')
            diffusers_name = diffusers_name.replace('proj.2', 'ff.net.2')
            diffusers_name = diffusers_name.replace('proj.3', 'norm')
            updated_state_dict[diffusers_name] = value
    elif 'perceiver_resampler.proj_in.weight' in state_dict:
        id_embeddings_dim = state_dict['proj.0.weight'].shape[1]
        embed_dims = state_dict['perceiver_resampler.proj_in.weight'].shape[0]
        hidden_dims = state_dict['perceiver_resampler.proj_in.weight'].shape[1]
        output_dims = state_dict['perceiver_resampler.proj_out.weight'].shape[0
            ]
        heads = state_dict['perceiver_resampler.layers.0.0.to_q.weight'].shape[
            0] // 64
        with init_context():
            image_projection = IPAdapterFaceIDPlusImageProjection(embed_dims
                =embed_dims, output_dims=output_dims, hidden_dims=
                hidden_dims, heads=heads, id_embeddings_dim=id_embeddings_dim)
        for key, value in state_dict.items():
            diffusers_name = key.replace('perceiver_resampler.', '')
            diffusers_name = diffusers_name.replace('0.to', 'attn.to')
            diffusers_name = diffusers_name.replace('0.1.0.', '0.ff.0.')
            diffusers_name = diffusers_name.replace('0.1.1.weight',
                '0.ff.1.net.0.proj.weight')
            diffusers_name = diffusers_name.replace('0.1.3.weight',
                '0.ff.1.net.2.weight')
            diffusers_name = diffusers_name.replace('1.1.0.', '1.ff.0.')
            diffusers_name = diffusers_name.replace('1.1.1.weight',
                '1.ff.1.net.0.proj.weight')
            diffusers_name = diffusers_name.replace('1.1.3.weight',
                '1.ff.1.net.2.weight')
            diffusers_name = diffusers_name.replace('2.1.0.', '2.ff.0.')
            diffusers_name = diffusers_name.replace('2.1.1.weight',
                '2.ff.1.net.0.proj.weight')
            diffusers_name = diffusers_name.replace('2.1.3.weight',
                '2.ff.1.net.2.weight')
            diffusers_name = diffusers_name.replace('3.1.0.', '3.ff.0.')
            diffusers_name = diffusers_name.replace('3.1.1.weight',
                '3.ff.1.net.0.proj.weight')
            diffusers_name = diffusers_name.replace('3.1.3.weight',
                '3.ff.1.net.2.weight')
            diffusers_name = diffusers_name.replace('layers.0.0',
                'layers.0.ln0')
            diffusers_name = diffusers_name.replace('layers.0.1',
                'layers.0.ln1')
            diffusers_name = diffusers_name.replace('layers.1.0',
                'layers.1.ln0')
            diffusers_name = diffusers_name.replace('layers.1.1',
                'layers.1.ln1')
            diffusers_name = diffusers_name.replace('layers.2.0',
                'layers.2.ln0')
            diffusers_name = diffusers_name.replace('layers.2.1',
                'layers.2.ln1')
            diffusers_name = diffusers_name.replace('layers.3.0',
                'layers.3.ln0')
            diffusers_name = diffusers_name.replace('layers.3.1',
                'layers.3.ln1')
            if 'norm1' in diffusers_name:
                updated_state_dict[diffusers_name.replace('0.norm1', '0')
                    ] = value
            elif 'norm2' in diffusers_name:
                updated_state_dict[diffusers_name.replace('0.norm2', '1')
                    ] = value
            elif 'to_kv' in diffusers_name:
                v_chunk = value.chunk(2, dim=0)
                updated_state_dict[diffusers_name.replace('to_kv', 'to_k')
                    ] = v_chunk[0]
                updated_state_dict[diffusers_name.replace('to_kv', 'to_v')
                    ] = v_chunk[1]
            elif 'to_out' in diffusers_name:
                updated_state_dict[diffusers_name.replace('to_out', 'to_out.0')
                    ] = value
            elif 'proj.0.weight' == diffusers_name:
                updated_state_dict['proj.net.0.proj.weight'] = value
            elif 'proj.0.bias' == diffusers_name:
                updated_state_dict['proj.net.0.proj.bias'] = value
            elif 'proj.2.weight' == diffusers_name:
                updated_state_dict['proj.net.2.weight'] = value
            elif 'proj.2.bias' == diffusers_name:
                updated_state_dict['proj.net.2.bias'] = value
            else:
                updated_state_dict[diffusers_name] = value
    elif 'norm.weight' in state_dict:
        id_embeddings_dim_in = state_dict['proj.0.weight'].shape[1]
        id_embeddings_dim_out = state_dict['proj.0.weight'].shape[0]
        multiplier = id_embeddings_dim_out // id_embeddings_dim_in
        norm_layer = 'norm.weight'
        cross_attention_dim = state_dict[norm_layer].shape[0]
        num_tokens = state_dict['proj.2.weight'].shape[0
            ] // cross_attention_dim
        with init_context():
            image_projection = IPAdapterFaceIDImageProjection(
                cross_attention_dim=cross_attention_dim, image_embed_dim=
                id_embeddings_dim_in, mult=multiplier, num_tokens=num_tokens)
        for key, value in state_dict.items():
            diffusers_name = key.replace('proj.0', 'ff.net.0.proj')
            diffusers_name = diffusers_name.replace('proj.2', 'ff.net.2')
            updated_state_dict[diffusers_name] = value
    else:
        num_image_text_embeds = state_dict['latents'].shape[1]
        embed_dims = state_dict['proj_in.weight'].shape[1]
        output_dims = state_dict['proj_out.weight'].shape[0]
        hidden_dims = state_dict['latents'].shape[2]
        heads = state_dict['layers.0.0.to_q.weight'].shape[0] // 64
        with init_context():
            image_projection = IPAdapterPlusImageProjection(embed_dims=
                embed_dims, output_dims=output_dims, hidden_dims=
                hidden_dims, heads=heads, num_queries=num_image_text_embeds)
        for key, value in state_dict.items():
            diffusers_name = key.replace('0.to', '2.to')
            diffusers_name = diffusers_name.replace('1.0.weight', '3.0.weight')
            diffusers_name = diffusers_name.replace('1.0.bias', '3.0.bias')
            diffusers_name = diffusers_name.replace('1.1.weight',
                '3.1.net.0.proj.weight')
            diffusers_name = diffusers_name.replace('1.3.weight',
                '3.1.net.2.weight')
            if 'norm1' in diffusers_name:
                updated_state_dict[diffusers_name.replace('0.norm1', '0')
                    ] = value
            elif 'norm2' in diffusers_name:
                updated_state_dict[diffusers_name.replace('0.norm2', '1')
                    ] = value
            elif 'to_kv' in diffusers_name:
                v_chunk = value.chunk(2, dim=0)
                updated_state_dict[diffusers_name.replace('to_kv', 'to_k')
                    ] = v_chunk[0]
                updated_state_dict[diffusers_name.replace('to_kv', 'to_v')
                    ] = v_chunk[1]
            elif 'to_out' in diffusers_name:
                updated_state_dict[diffusers_name.replace('to_out', 'to_out.0')
                    ] = value
            else:
                updated_state_dict[diffusers_name] = value
    if not low_cpu_mem_usage:
        image_projection.load_state_dict(updated_state_dict)
    else:
        load_model_dict_into_meta(image_projection, updated_state_dict,
            device=self.device, dtype=self.dtype)
    return image_projection
