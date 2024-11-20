def __init__(self, bboxes, segments=None, keypoints=None, bbox_format=
    'xywh', normalized=True) ->None:
    """
        Args:
            bboxes (ndarray): bboxes with shape [N, 4].
            segments (list | ndarray): segments.
            keypoints (ndarray): keypoints(x, y, visible) with shape [N, 17, 3].
        """
    if segments is None:
        segments = []
    self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
    self.keypoints = keypoints
    self.normalized = normalized
    if len(segments) > 0:
        segments = resample_segments(segments)
        segments = np.stack(segments, axis=0)
    else:
        segments = np.zeros((0, 1000, 2), dtype=np.float32)
    self.segments = segments
