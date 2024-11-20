def _prepare_pred(self, pred, pbatch, proto):
    """Prepares a batch for training or inference by processing images and targets."""
    predn = super()._prepare_pred(pred, pbatch)
    pred_masks = self.process(proto, pred[:, 6:], pred[:, :4], shape=pbatch
        ['imgsz'])
    return predn, pred_masks
