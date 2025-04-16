def calculate_metrics(self, batch):
    inputs, labels = batch
    logits = self.model(inputs)
    logits[inputs != 0] = -float('Inf')
    metrics = recalls_and_ndcgs_for_ks(logits, labels, self.metric_ks)
    return metrics
