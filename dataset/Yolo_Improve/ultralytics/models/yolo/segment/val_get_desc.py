def get_desc(self):
    """Return a formatted description of evaluation metrics."""
    return ('%22s' + '%11s' * 10) % ('Class', 'Images', 'Instances',
        'Box(P', 'R', 'mAP50', 'mAP50-95)', 'Mask(P', 'R', 'mAP50', 'mAP50-95)'
        )
