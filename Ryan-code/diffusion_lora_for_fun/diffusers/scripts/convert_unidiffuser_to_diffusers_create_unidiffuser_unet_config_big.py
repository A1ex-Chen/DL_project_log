def create_unidiffuser_unet_config_big():
    unet_config = {'text_dim': 64, 'clip_img_dim': 512, 'num_text_tokens': 
        77, 'num_attention_heads': 24, 'attention_head_dim': 64,
        'in_channels': 4, 'out_channels': 4, 'num_layers': 30, 'dropout': 
        0.0, 'norm_num_groups': 32, 'attention_bias': False, 'sample_size':
        64, 'patch_size': 2, 'activation_fn': 'gelu', 'num_embeds_ada_norm':
        1000, 'norm_type': 'layer_norm', 'block_type': 'unidiffuser',
        'pre_layer_norm': False, 'use_timestep_embedding': False,
        'norm_elementwise_affine': True, 'use_patch_pos_embed': False,
        'ff_final_dropout': True, 'use_data_type_embedding': False}
    return unet_config
