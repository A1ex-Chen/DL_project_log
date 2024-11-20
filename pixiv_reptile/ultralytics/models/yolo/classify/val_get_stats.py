def get_stats(self):
    """Returns a dictionary of metrics obtained by processing targets and predictions."""
    self.metrics.process(self.targets, self.pred)
    return self.metrics.results_dict
