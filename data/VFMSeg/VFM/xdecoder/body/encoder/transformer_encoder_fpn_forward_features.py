def forward_features(self, features):
    multi_scale_features = []
    num_cur_levels = 0
    for idx, f in enumerate(self.in_features[::-1]):
        x = features[f]
        lateral_conv = self.lateral_convs[idx]
        output_conv = self.output_convs[idx]
        if lateral_conv is None:
            transformer = self.input_proj(x)
            pos = self.pe_layer(x)
            transformer = self.transformer(transformer, None, pos)
            y = output_conv(transformer)
            transformer_encoder_features = transformer
        else:
            cur_fpn = lateral_conv(x)
            y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode=
                'nearest')
            y = output_conv(y)
        if num_cur_levels < self.maskformer_num_feature_levels:
            multi_scale_features.append(y)
            num_cur_levels += 1
    mask_features = self.mask_features(y) if self.mask_on else None
    return mask_features, transformer_encoder_features, multi_scale_features
