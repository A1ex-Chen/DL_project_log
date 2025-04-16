@property
def polygons(self):
    if self._polygons is None:
        self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
    return self._polygons
