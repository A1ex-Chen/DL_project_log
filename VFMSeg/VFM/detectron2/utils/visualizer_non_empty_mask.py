def non_empty_mask(self):
    """
        Returns:
            (H, W) array, a mask for all pixels that have a prediction
        """
    empty_ids = []
    for id in self._seg_ids:
        if id not in self._sinfo:
            empty_ids.append(id)
    if len(empty_ids) == 0:
        return np.zeros(self._seg.shape, dtype=np.uint8)
    assert len(empty_ids
        ) == 1, '>1 ids corresponds to no labels. This is currently not supported'
    return (self._seg != empty_ids[0]).numpy().astype(np.bool)
