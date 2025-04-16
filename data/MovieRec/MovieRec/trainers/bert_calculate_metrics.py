def calculate_metrics(self, batch):
    seqs, candidates, labels = batch
    scores = self.model(seqs)
    scores = scores[:, -1, :]
    print(scores.argmax(dim=-1))
    scores = scores.gather(1, candidates)
    metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
    return metrics
