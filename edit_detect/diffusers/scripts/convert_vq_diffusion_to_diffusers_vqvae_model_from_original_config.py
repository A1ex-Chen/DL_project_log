def vqvae_model_from_original_config(original_config):
    assert original_config['target'
        ] in PORTED_VQVAES, f"{original_config['target']} has not yet been ported to diffusers."
    original_config = original_config['params']
    original_encoder_config = original_config['encoder_config']['params']
    original_decoder_config = original_config['decoder_config']['params']
    in_channels = original_encoder_config['in_channels']
    out_channels = original_decoder_config['out_ch']
    down_block_types = get_down_block_types(original_encoder_config)
    up_block_types = get_up_block_types(original_decoder_config)
    assert original_encoder_config['ch'] == original_decoder_config['ch']
    assert original_encoder_config['ch_mult'] == original_decoder_config[
        'ch_mult']
    block_out_channels = tuple([(original_encoder_config['ch'] * a_ch_mult) for
        a_ch_mult in original_encoder_config['ch_mult']])
    assert original_encoder_config['num_res_blocks'
        ] == original_decoder_config['num_res_blocks']
    layers_per_block = original_encoder_config['num_res_blocks']
    assert original_encoder_config['z_channels'] == original_decoder_config[
        'z_channels']
    latent_channels = original_encoder_config['z_channels']
    num_vq_embeddings = original_config['n_embed']
    norm_num_groups = 32
    e_dim = original_config['embed_dim']
    model = VQModel(in_channels=in_channels, out_channels=out_channels,
        down_block_types=down_block_types, up_block_types=up_block_types,
        block_out_channels=block_out_channels, layers_per_block=
        layers_per_block, latent_channels=latent_channels,
        num_vq_embeddings=num_vq_embeddings, norm_num_groups=
        norm_num_groups, vq_embed_dim=e_dim)
    return model
