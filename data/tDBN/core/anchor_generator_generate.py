def generate(self, feature_map_size):
    return box_np_ops.create_anchors_3d_range(feature_map_size, self.
        _anchor_ranges, self._sizes, self._rotations, self._dtype)
