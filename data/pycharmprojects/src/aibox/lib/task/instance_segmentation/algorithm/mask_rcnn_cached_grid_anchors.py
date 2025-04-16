def cached_grid_anchors(self, grid_sizes, strides):
    key = str(grid_sizes + strides)
    anchors = self.grid_anchors(grid_sizes, strides)
    self._cache[key] = anchors
    return anchors
