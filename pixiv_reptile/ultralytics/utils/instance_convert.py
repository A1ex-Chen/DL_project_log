def convert(self, format):
    """Converts bounding box format from one type to another."""
    assert format in _formats, f'Invalid bounding box format: {format}, format must be one of {_formats}'
    if self.format == format:
        return
    elif self.format == 'xyxy':
        func = xyxy2xywh if format == 'xywh' else xyxy2ltwh
    elif self.format == 'xywh':
        func = xywh2xyxy if format == 'xyxy' else xywh2ltwh
    else:
        func = ltwh2xyxy if format == 'xyxy' else ltwh2xywh
    self.bboxes = func(self.bboxes)
    self.format = format
