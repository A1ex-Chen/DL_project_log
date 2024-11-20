def get_dummy_components(self):
    torch.manual_seed(0)
    transformer = Transformer2DModel(sample_size=16, num_layers=2,
        patch_size=4, attention_head_dim=8, num_attention_heads=2,
        in_channels=4, out_channels=8, attention_bias=True, activation_fn=
        'gelu-approximate', num_embeds_ada_norm=1000, norm_type=
        'ada_norm_zero', norm_elementwise_affine=False)
    vae = AutoencoderKL()
    scheduler = DDIMScheduler()
    components = {'transformer': transformer.eval(), 'vae': vae.eval(),
        'scheduler': scheduler}
    return components
