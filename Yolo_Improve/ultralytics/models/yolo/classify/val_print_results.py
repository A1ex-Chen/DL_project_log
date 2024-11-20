def print_results(self):
    """Prints evaluation metrics for YOLO object detection model."""
    pf = '%22s' + '%11.3g' * len(self.metrics.keys)
    LOGGER.info(pf % ('all', self.metrics.top1, self.metrics.top5))
