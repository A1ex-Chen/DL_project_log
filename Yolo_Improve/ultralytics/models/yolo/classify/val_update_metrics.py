def update_metrics(self, preds, batch):
    """Updates running metrics with model predictions and batch targets."""
    n5 = min(len(self.names), 5)
    self.pred.append(preds.argsort(1, descending=True)[:, :n5].type(torch.
        int32).cpu())
    self.targets.append(batch['cls'].type(torch.int32).cpu())
