def edgevit_xs(**kwargs):
    model = EdgeVit(depth=[1, 1, 3, 1], embed_dim=[48, 96, 240, 384],
        head_dim=48, mlp_ratio=[4] * 4, qkv_bias=True, norm_layer=partial(
        nn.LayerNorm, eps=1e-06), sr_ratios=[4, 2, 2, 1], **kwargs)
    return model
