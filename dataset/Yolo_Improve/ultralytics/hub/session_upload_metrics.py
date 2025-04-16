def upload_metrics(self):
    """Upload model metrics to Ultralytics HUB."""
    return self.request_queue(self.model.upload_metrics, metrics=self.
        metrics_queue.copy(), thread=True)
