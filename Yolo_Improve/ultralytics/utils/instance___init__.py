def __init__(self, bboxes, segments=None, keypoints=None, bbox_format=
    'xywh', normalized=True) ->None:
    """
        Args:
            bboxes (ndarray): bboxes with shape [N, 4].
            segments (list | ndarray): segments.
            keypoints (ndarray): keypoints(x, y, visible) with shape [N, 17, 3].
        """
    self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
    self.keypoints = keypoints
    self.normalized = normalized
    self.segments = segments
