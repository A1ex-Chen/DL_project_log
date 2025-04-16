def mean_results(self):
    """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
    return self.box.mean_results()
