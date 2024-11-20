@property
def has_holes(self):
    if self._has_holes is None:
        if self._mask is not None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        else:
            self._has_holes = False
    return self._has_holes
