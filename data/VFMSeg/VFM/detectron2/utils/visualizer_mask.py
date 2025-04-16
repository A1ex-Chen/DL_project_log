@property
def mask(self):
    if self._mask is None:
        self._mask = self.polygons_to_mask(self._polygons)
    return self._mask
