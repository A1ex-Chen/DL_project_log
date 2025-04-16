def output_shape(self):
    """
        Returns:
            dict[str->ShapeSpec]
        """
    return {name: ShapeSpec(channels=self._out_feature_channels[name],
        stride=self._out_feature_strides[name]) for name in self._out_features}
