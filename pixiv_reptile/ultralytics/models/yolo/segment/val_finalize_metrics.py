def finalize_metrics(self, *args, **kwargs):
    """Sets speed and confusion matrix for evaluation metrics."""
    self.metrics.speed = self.speed
    self.metrics.confusion_matrix = self.confusion_matrix
