def get_dummy_components(self):
    torch.manual_seed(0)
    transformer = Transformer2DModel(sample_size=8, num_layers=2,
        patch_size=2, attention_head_dim=8, num_attention_heads=3,
        caption_channels=32, in_channels=4, cross_attention_dim=24,
        out_channels=8, attention_bias=True, activation_fn=
        'gelu-approximate', num_embeds_ada_norm=1000, norm_type=
        'ada_norm_single', norm_elementwise_affine=False, norm_eps=1e-06)
    torch.manual_seed(0)
    vae = AutoencoderKL()
    scheduler = DDIMScheduler()
    text_encoder = T5EncoderModel.from_pretrained(
        'hf-internal-testing/tiny-random-t5')
    tokenizer = AutoTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-t5')
    components = {'transformer': transformer.eval(), 'vae': vae.eval(),
        'scheduler': scheduler, 'text_encoder': text_encoder, 'tokenizer':
        tokenizer}
    return components
