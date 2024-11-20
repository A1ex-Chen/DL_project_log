def clip(self, w, h):
    """Clips bounding boxes, segments, and keypoints values to stay within image boundaries."""
    ori_format = self._bboxes.format
    self.convert_bbox(format='xyxy')
    self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
    self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
    if ori_format != 'xyxy':
        self.convert_bbox(format=ori_format)
    self.segments[..., 0] = self.segments[..., 0].clip(0, w)
    self.segments[..., 1] = self.segments[..., 1].clip(0, h)
    if self.keypoints is not None:
        self.keypoints[..., 0] = self.keypoints[..., 0].clip(0, w)
        self.keypoints[..., 1] = self.keypoints[..., 1].clip(0, h)
