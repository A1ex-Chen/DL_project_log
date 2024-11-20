def compute_loss(self, predictions, targets):
    self.loss += ((predictions - targets) ** 2).sum((1, 2)).mean(0)
