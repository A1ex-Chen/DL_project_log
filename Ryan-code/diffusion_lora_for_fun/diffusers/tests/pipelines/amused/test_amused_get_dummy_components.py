def get_dummy_components(self):
    torch.manual_seed(0)
    transformer = UVit2DModel(hidden_size=8, use_bias=False, hidden_dropout
        =0.0, cond_embed_dim=8, micro_cond_encode_dim=2,
        micro_cond_embed_dim=10, encoder_hidden_size=8, vocab_size=32,
        codebook_size=8, in_channels=8, block_out_channels=8,
        num_res_blocks=1, downsample=True, upsample=True, block_num_heads=1,
        num_hidden_layers=1, num_attention_heads=1, attention_dropout=0.0,
        intermediate_size=8, layer_norm_eps=1e-06, ln_elementwise_affine=True)
    scheduler = AmusedScheduler(mask_token_id=31)
    torch.manual_seed(0)
    vqvae = VQModel(act_fn='silu', block_out_channels=[8], down_block_types
        =['DownEncoderBlock2D'], in_channels=3, latent_channels=8,
        layers_per_block=1, norm_num_groups=8, num_vq_embeddings=8,
        out_channels=3, sample_size=8, up_block_types=['UpDecoderBlock2D'],
        mid_block_add_attention=False, lookup_from_codebook=True)
    torch.manual_seed(0)
    text_encoder_config = CLIPTextConfig(bos_token_id=0, eos_token_id=2,
        hidden_size=8, intermediate_size=8, layer_norm_eps=1e-05,
        num_attention_heads=1, num_hidden_layers=1, pad_token_id=1,
        vocab_size=1000, projection_dim=8)
    text_encoder = CLIPTextModelWithProjection(text_encoder_config)
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    components = {'transformer': transformer, 'scheduler': scheduler,
        'vqvae': vqvae, 'text_encoder': text_encoder, 'tokenizer': tokenizer}
    return components
