def _update_labels(self, labels, ratio, padw, padh):
    """Update labels."""
    labels['instances'].convert_bbox(format='xyxy')
    labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
    labels['instances'].scale(*ratio)
    labels['instances'].add_padding(padw, padh)
    return labels
