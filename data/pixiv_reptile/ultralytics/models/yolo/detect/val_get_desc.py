def get_desc(self):
    """Return a formatted string summarizing class metrics of YOLO model."""
    return ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'Box(P',
        'R', 'mAP50', 'mAP50-95)')
