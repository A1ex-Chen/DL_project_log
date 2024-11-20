def loss(self, batch, preds=None):
    if not hasattr(self, 'criterion'):
        self.criterion = v8DetectionLoss(self)
    preds = self._forward(batch['img']) if preds is None else preds
    return self.criterion(preds, batch)
