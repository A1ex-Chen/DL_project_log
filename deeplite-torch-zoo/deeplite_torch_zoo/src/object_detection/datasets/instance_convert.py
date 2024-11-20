def convert(self, format):
    """Converts bounding box format from one type to another."""
    assert format in _formats, f'Invalid bounding box format: {format}, format must be one of {_formats}'
    if self.format == format:
        return
    elif self.format == 'xyxy':
        bboxes = xyxy2xywh(self.bboxes) if format == 'xywh' else xyxy2ltwh(self
            .bboxes)
    elif self.format == 'xywh':
        bboxes = xywh2xyxy(self.bboxes) if format == 'xyxy' else xywh2ltwh(self
            .bboxes)
    else:
        bboxes = ltwh2xyxy(self.bboxes) if format == 'xyxy' else ltwh2xywh(self
            .bboxes)
    self.bboxes = bboxes
    self.format = format
