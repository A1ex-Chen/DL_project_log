def __repr__(self):
    feature_str = str(self._feature
        ) if self._feature is None else f'array of shape {self._feature.shape}'
    return (
        f'Detection(box={self._box}, score={self._score:.5f}, class_id={self._class_id}, feature={feature_str})'
        )
