def evaluate_mertics(self, pred, target):
    metric_dict = {}
    for i in range(len(self.metric_names)):
        metric_dict[self.metric_names[i]] = self.metrics[i](pred, target)
    return metric_dict
