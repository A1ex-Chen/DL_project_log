@property
def keys(self):
    """Returns a list of keys for accessing specific metrics."""
    return ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)',
        'metrics/mAP50-95(B)']
