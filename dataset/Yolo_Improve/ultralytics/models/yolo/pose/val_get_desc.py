def get_desc(self):
    """Returns description of evaluation metrics in string format."""
    return ('%22s' + '%11s' * 10) % ('Class', 'Images', 'Instances',
        'Box(P', 'R', 'mAP50', 'mAP50-95)', 'Pose(P', 'R', 'mAP50', 'mAP50-95)'
        )
