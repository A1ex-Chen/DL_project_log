@autocast(enabled=False)
def forward_features(self, features):
    srcs = []
    pos = []
    for idx, f in enumerate(self.transformer_in_features[::-1]):
        x = features[f].float()
        srcs.append(self.input_proj[idx](x))
        pos.append(self.pe_layer(x))
    y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
    bs = y.shape[0]
    split_size_or_sections = [None] * self.transformer_num_feature_levels
    for i in range(self.transformer_num_feature_levels):
        if i < self.transformer_num_feature_levels - 1:
            split_size_or_sections[i] = level_start_index[i + 1
                ] - level_start_index[i]
        else:
            split_size_or_sections[i] = y.shape[1] - level_start_index[i]
    y = torch.split(y, split_size_or_sections, dim=1)
    out = []
    multi_scale_features = []
    num_cur_levels = 0
    for i, z in enumerate(y):
        out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0],
            spatial_shapes[i][1]))
    for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
        x = features[f].float()
        lateral_conv = self.lateral_convs[idx]
        output_conv = self.output_convs[idx]
        cur_fpn = lateral_conv(x)
        y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode=
            'bilinear', align_corners=False)
        y = output_conv(y)
        out.append(y)
    for o in out:
        if num_cur_levels < self.maskformer_num_feature_levels:
            multi_scale_features.append(o)
            num_cur_levels += 1
    return self.mask_features(out[-1]), out[0], multi_scale_features
