def log_metric(self, metric_name, val, n=1):
    self.metrics[metric_name]['meter'].record(val, n=n)
