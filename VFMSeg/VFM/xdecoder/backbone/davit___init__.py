def __init__(self, cfg, input_shape):
    spec = cfg['BACKBONE']['DAVIT']
    super().__init__(num_classes=0, depths=spec['DEPTHS'], embed_dims=spec[
        'DIM_EMBED'], num_heads=spec['NUM_HEADS'], num_groups=spec[
        'NUM_GROUPS'], patch_size=spec['PATCH_SIZE'], patch_stride=spec[
        'PATCH_STRIDE'], patch_padding=spec['PATCH_PADDING'], patch_prenorm
        =spec['PATCH_PRENORM'], drop_path_rate=spec['DROP_PATH_RATE'],
        img_size=input_shape, window_size=spec.get('WINDOW_SIZE', 7),
        enable_checkpoint=spec.get('ENABLE_CHECKPOINT', False),
        conv_at_attn=spec.get('CONV_AT_ATTN', True), conv_at_ffn=spec.get(
        'CONV_AT_FFN', True), out_indices=spec.get('OUT_INDICES', []))
    self._out_features = cfg['BACKBONE']['DAVIT']['OUT_FEATURES']
    self._out_feature_strides = {'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32}
    self._out_feature_channels = {'res2': self.embed_dims[0], 'res3': self.
        embed_dims[1], 'res4': self.embed_dims[2], 'res5': self.embed_dims[3]}
