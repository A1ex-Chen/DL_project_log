@register_backbone
def get_swin_backbone(cfg):
    swin_cfg = cfg['MODEL']['BACKBONE']['SWIN']
    pretrain_img_size = swin_cfg['PRETRAIN_IMG_SIZE']
    patch_size = swin_cfg['PATCH_SIZE']
    in_chans = 3
    embed_dim = swin_cfg['EMBED_DIM']
    depths = swin_cfg['DEPTHS']
    num_heads = swin_cfg['NUM_HEADS']
    window_size = swin_cfg['WINDOW_SIZE']
    mlp_ratio = swin_cfg['MLP_RATIO']
    qkv_bias = swin_cfg['QKV_BIAS']
    qk_scale = swin_cfg['QK_SCALE']
    drop_rate = swin_cfg['DROP_RATE']
    attn_drop_rate = swin_cfg['ATTN_DROP_RATE']
    drop_path_rate = swin_cfg['DROP_PATH_RATE']
    norm_layer = nn.LayerNorm
    ape = swin_cfg['APE']
    patch_norm = swin_cfg['PATCH_NORM']
    use_checkpoint = swin_cfg['USE_CHECKPOINT']
    out_indices = swin_cfg.get('OUT_INDICES', [0, 1, 2, 3])
    swin = D2SwinTransformer(swin_cfg, pretrain_img_size, patch_size,
        in_chans, embed_dim, depths, num_heads, window_size, mlp_ratio,
        qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
        norm_layer, ape, patch_norm, out_indices, use_checkpoint=use_checkpoint
        )
    if cfg['MODEL']['BACKBONE']['LOAD_PRETRAINED'] is True:
        filename = cfg['MODEL']['BACKBONE']['PRETRAINED']
        with PathManager.open(filename, 'rb') as f:
            ckpt = torch.load(f, map_location=cfg['device'])['model']
        swin.load_weights(ckpt, swin_cfg.get('PRETRAINED_LAYERS', ['*']),
            cfg['VERBOSE'])
    return swin
