def forward(self, features):
    """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[RotatedBoxes]: a list of Boxes containing all the anchors for each feature map
                (i.e. the cell anchors repeated over all locations in the feature map).
                The number of anchors of each feature map is Hi x Wi x num_cell_anchors,
                where Hi, Wi are resolution of the feature map divided by anchor stride.
        """
    grid_sizes = [feature_map.shape[-2:] for feature_map in features]
    anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)
    return [RotatedBoxes(x) for x in anchors_over_all_feature_maps]
