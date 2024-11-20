def edgevit_xxs(**kwargs):
    model = EdgeVit(depth=[1, 1, 3, 2], embed_dim=[36, 72, 144, 288],
        head_dim=36, mlp_ratio=[4] * 4, qkv_bias=True, norm_layer=partial(
        nn.LayerNorm, eps=1e-06), sr_ratios=[4, 2, 2, 1], **kwargs)
    return model
