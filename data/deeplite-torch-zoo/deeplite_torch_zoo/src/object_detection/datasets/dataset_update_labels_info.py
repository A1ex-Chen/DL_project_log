def update_labels_info(self, label):
    """custom your label format here."""
    bboxes = label.pop('bboxes')
    segments = label.pop('segments')
    keypoints = label.pop('keypoints', None)
    bbox_format = label.pop('bbox_format')
    normalized = label.pop('normalized')
    label['instances'] = Instances(bboxes, segments, keypoints, bbox_format
        =bbox_format, normalized=normalized)
    return label
