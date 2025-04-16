def get_desc(self):
    """Returns a formatted string summarizing classification metrics."""
    return ('%22s' + '%11s' * 2) % ('classes', 'top1_acc', 'top5_acc')
