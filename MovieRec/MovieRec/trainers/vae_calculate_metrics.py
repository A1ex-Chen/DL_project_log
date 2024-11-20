def calculate_metrics(self, batch):
    inputs, labels = batch
    logits, _, _ = self.model(inputs)
    logits[inputs != 0] = -float('Inf')
    metrics = recalls_and_ndcgs_for_ks(logits, labels, self.metric_ks)
    if self.finding_best_beta:
        if self.current_best_metric < metrics[self.best_metric]:
            self.current_best_metric = metrics[self.best_metric]
            self.best_beta = self.__beta
    return metrics
