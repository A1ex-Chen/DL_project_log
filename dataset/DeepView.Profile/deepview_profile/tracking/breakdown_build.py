def build(self):
    if self._peak_usage_bytes is None:
        raise RuntimeError(
            'Missing peak usage when constructing the breakdown.')
    self._prune_tree(self._operation_root)
    self._prune_tree(self._weight_root)
    self._operation_root.build_context_info_map()
    return HierarchicalBreakdown(operations=self._operation_root, weights=
        self._weight_root, peak_usage_bytes=self._peak_usage_bytes)
