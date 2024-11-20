def text_proj_from_original_config():
    time_embed_dim = DECODER_CONFIG['block_out_channels'][0] * 4
    cross_attention_dim = DECODER_CONFIG['cross_attention_dim']
    model = UnCLIPTextProjModel(time_embed_dim=time_embed_dim,
        cross_attention_dim=cross_attention_dim)
    return model
