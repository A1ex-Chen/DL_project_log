def mean_results(self):
    return self.metric_box.mean_results() + self.metric_mask.mean_results()
