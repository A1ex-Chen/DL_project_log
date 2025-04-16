def loss(self, data, target, reduce='mean'):
    """Calculate a value of loss function"""
    logits = self(data)
    for task, logit in logits.items():
        logits[task] = logit.to(self.device)
    losses = {}
    for task, label in target.items():
        label = label.to(self.device)
        losses[task] = self.criterion(logits[task], label)
    if reduce:
        total = 0
        for _, value in losses.items():
            total += value
        if reduce == 'mean':
            losses = total / len(losses)
        elif reduce == 'sum':
            losses = total
        else:
            raise ValueError('Reduced loss must use either `mean` or `sum`!')
    return losses
