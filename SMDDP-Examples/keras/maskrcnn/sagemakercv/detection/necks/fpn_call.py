def call(self, inputs, *args, **kwargs):
    feats_bottom_up = inputs
    feats_lateral = {}
    for level in range(self._min_level, self._upsample_max_level + 1):
        feats_lateral[level] = self._local_layers['stage1'][level](
            feats_bottom_up[level])
    feats = {self._upsample_max_level: feats_lateral[self._upsample_max_level]}
    for level in range(self._upsample_max_level - 1, self._min_level - 1, -1):
        feats[level] = spatial_transform_ops.nearest_upsampling(feats[level +
            1], 2) + feats_lateral[level]
    for level in range(self._min_level, self._upsample_max_level + 1):
        feats[level] = self._local_layers['stage2'][level](feats[level])
    if self._max_level == self._upsample_max_level + 1:
        feats[self._max_level] = self._local_layers['stage3_1'](feats[self.
            _max_level - 1])
    else:
        for level in range(self._upsample_max_level + 1, self._max_level + 1):
            feats[level] = self._local_layers['stage3_2'][level](feats[
                level - 1])
    return feats
