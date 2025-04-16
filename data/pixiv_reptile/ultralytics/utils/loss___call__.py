def __call__(self, preds, batch):
    """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
    preds = preds[1] if isinstance(preds, tuple) else preds
    one2many = preds['one2many']
    loss_one2many = self.one2many(one2many, batch)
    one2one = preds['one2one']
    loss_one2one = self.one2one(one2one, batch)
    return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[
        1]
