def __init__(self, cfg, pretrain_img_size, patch_size, in_chans, embed_dim,
    depths, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale,
    drop_rate, attn_drop_rate, drop_path_rate, norm_layer, ape, patch_norm,
    out_indices, use_checkpoint):
    super().__init__(pretrain_img_size, patch_size, in_chans, embed_dim,
        depths, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale,
        drop_rate, attn_drop_rate, drop_path_rate, norm_layer, ape,
        patch_norm, out_indices, use_checkpoint=use_checkpoint)
    self._out_features = cfg['OUT_FEATURES']
    self._out_feature_strides = {'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32}
    self._out_feature_channels = {'res2': self.num_features[0], 'res3':
        self.num_features[1], 'res4': self.num_features[2], 'res5': self.
        num_features[3]}
