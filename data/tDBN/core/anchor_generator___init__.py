def __init__(self, anchor_ranges, sizes=[1.6, 3.9, 1.56], rotations=[0, np.
    pi / 2], class_id=None, match_threshold=-1, unmatch_threshold=-1, dtype
    =np.float32):
    self._sizes = sizes
    self._anchor_ranges = anchor_ranges
    self._rotations = rotations
    self._dtype = dtype
    self._class_id = class_id
    self._match_threshold = match_threshold
    self._unmatch_threshold = unmatch_threshold
