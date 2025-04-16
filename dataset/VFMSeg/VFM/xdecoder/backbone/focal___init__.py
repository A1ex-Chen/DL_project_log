def __init__(self, cfg, input_shape):
    pretrain_img_size = cfg['BACKBONE']['FOCAL']['PRETRAIN_IMG_SIZE']
    patch_size = cfg['BACKBONE']['FOCAL']['PATCH_SIZE']
    in_chans = 3
    embed_dim = cfg['BACKBONE']['FOCAL']['EMBED_DIM']
    depths = cfg['BACKBONE']['FOCAL']['DEPTHS']
    mlp_ratio = cfg['BACKBONE']['FOCAL']['MLP_RATIO']
    drop_rate = cfg['BACKBONE']['FOCAL']['DROP_RATE']
    drop_path_rate = cfg['BACKBONE']['FOCAL']['DROP_PATH_RATE']
    norm_layer = nn.LayerNorm
    patch_norm = cfg['BACKBONE']['FOCAL']['PATCH_NORM']
    use_checkpoint = cfg['BACKBONE']['FOCAL']['USE_CHECKPOINT']
    out_indices = cfg['BACKBONE']['FOCAL']['OUT_INDICES']
    scaling_modulator = cfg['BACKBONE']['FOCAL'].get('SCALING_MODULATOR', False
        )
    super().__init__(pretrain_img_size, patch_size, in_chans, embed_dim,
        depths, mlp_ratio, drop_rate, drop_path_rate, norm_layer,
        patch_norm, out_indices, focal_levels=cfg['BACKBONE']['FOCAL'][
        'FOCAL_LEVELS'], focal_windows=cfg['BACKBONE']['FOCAL'][
        'FOCAL_WINDOWS'], use_conv_embed=cfg['BACKBONE']['FOCAL'][
        'USE_CONV_EMBED'], use_postln=cfg['BACKBONE']['FOCAL']['USE_POSTLN'
        ], use_postln_in_modulation=cfg['BACKBONE']['FOCAL'][
        'USE_POSTLN_IN_MODULATION'], scaling_modulator=scaling_modulator,
        use_layerscale=cfg['BACKBONE']['FOCAL']['USE_LAYERSCALE'],
        use_checkpoint=use_checkpoint)
    self._out_features = cfg['BACKBONE']['FOCAL']['OUT_FEATURES']
    self._out_feature_strides = {'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32}
    self._out_feature_channels = {'res2': self.num_features[0], 'res3':
        self.num_features[1], 'res4': self.num_features[2], 'res5': self.
        num_features[3]}
