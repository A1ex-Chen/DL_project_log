def __str__(self):
    overall_ret = self.compute_metrics()
    return self.metrics_to_str(overall_ret)
