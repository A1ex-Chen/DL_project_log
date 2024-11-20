def forward(self, features):
    grid_sizes = [feature_map.shape[-2:] for feature_map in features]
    anchors_over_all_feature_maps, centers_over_all_feature_maps = (self.
        _grid_anchors(grid_sizes))
    anchor_boxes = [Boxes(x) for x in anchors_over_all_feature_maps]
    return anchor_boxes, centers_over_all_feature_maps
