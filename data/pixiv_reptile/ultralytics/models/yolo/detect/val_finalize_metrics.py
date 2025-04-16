def finalize_metrics(self, *args, **kwargs):
    """Set final values for metrics speed and confusion matrix."""
    self.metrics.speed = self.speed
    self.metrics.confusion_matrix = self.confusion_matrix
