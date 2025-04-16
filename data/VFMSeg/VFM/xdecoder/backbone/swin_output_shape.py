def output_shape(self):
    feature_names = list(set(self._out_feature_strides.keys()) & set(self.
        _out_features))
    return {name: ShapeSpec(channels=self._out_feature_channels[name],
        stride=self._out_feature_strides[name]) for name in feature_names}
