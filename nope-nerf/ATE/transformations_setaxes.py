def setaxes(self, *axes):
    """Set axes to constrain rotations."""
    if axes is None:
        self._axes = None
    else:
        self._axes = [unit_vector(axis) for axis in axes]
