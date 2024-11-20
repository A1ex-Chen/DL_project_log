def __eq__(self, other):
    if not isinstance(other, Detection):
        return NotImplemented
    return np.array_equal(self._box, other.box) and np.isclose(self._score,
        other.score, atol=1e-05
        ) and self._class_id == other.class_id and np.array_equal(self.
        _feature, other.feature)
