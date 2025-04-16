def transformer_model_from_original_config(original_diffusion_config,
    original_transformer_config, original_content_embedding_config):
    assert original_diffusion_config['target'
        ] in PORTED_DIFFUSIONS, f"{original_diffusion_config['target']} has not yet been ported to diffusers."
    assert original_transformer_config['target'
        ] in PORTED_TRANSFORMERS, f"{original_transformer_config['target']} has not yet been ported to diffusers."
    assert original_content_embedding_config['target'
        ] in PORTED_CONTENT_EMBEDDINGS, f"{original_content_embedding_config['target']} has not yet been ported to diffusers."
    original_diffusion_config = original_diffusion_config['params']
    original_transformer_config = original_transformer_config['params']
    original_content_embedding_config = original_content_embedding_config[
        'params']
    inner_dim = original_transformer_config['n_embd']
    n_heads = original_transformer_config['n_head']
    assert inner_dim % n_heads == 0
    d_head = inner_dim // n_heads
    depth = original_transformer_config['n_layer']
    context_dim = original_transformer_config['condition_dim']
    num_embed = original_content_embedding_config['num_embed']
    num_embed = num_embed + 1
    height = original_transformer_config['content_spatial_size'][0]
    width = original_transformer_config['content_spatial_size'][1]
    assert width == height, 'width has to be equal to height'
    dropout = original_transformer_config['resid_pdrop']
    num_embeds_ada_norm = original_diffusion_config['diffusion_step']
    model_kwargs = {'attention_bias': True, 'cross_attention_dim':
        context_dim, 'attention_head_dim': d_head, 'num_layers': depth,
        'dropout': dropout, 'num_attention_heads': n_heads,
        'num_vector_embeds': num_embed, 'num_embeds_ada_norm':
        num_embeds_ada_norm, 'norm_num_groups': 32, 'sample_size': width,
        'activation_fn': 'geglu-approximate'}
    model = Transformer2DModel(**model_kwargs)
    return model
