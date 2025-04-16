def create_eva_vit_g(img_size=224, drop_path_rate=0.4, use_checkpoint=False,
    precision='fp16'):
    model = VisionTransformer(img_size=img_size, patch_size=14,
        use_mean_pooling=False, embed_dim=1408, depth=39, num_heads=1408 //
        88, mlp_ratio=4.3637, qkv_bias=True, drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-06), use_checkpoint=
        use_checkpoint)
    url = (
        'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth'
        )
    cached_file = download_cached_file(url, check_hash=False, progress=True)
    state_dict = torch.load(cached_file, map_location='cpu')
    interpolate_pos_embed(model, state_dict)
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    if precision == 'fp16':
        convert_weights_to_fp16(model)
    return model
