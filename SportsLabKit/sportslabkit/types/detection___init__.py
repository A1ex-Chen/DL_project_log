def __init__(self, box: Box, score: (float | None)=None, class_id: (int |
    None)=None, feature: (Vector | None)=None):
    box = np.array(box).squeeze()
    if box.shape != (4,):
        raise ValueError(
            f'box should have the shape (4, ), but got {box.shape}')
    self._box = box
    self._score = score
    self._class_id = class_id
    self._feature = feature
