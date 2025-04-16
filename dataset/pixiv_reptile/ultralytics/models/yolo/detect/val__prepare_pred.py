def _prepare_pred(self, pred, pbatch):
    """Prepares a batch of images and annotations for validation."""
    predn = pred.clone()
    ops.scale_boxes(pbatch['imgsz'], predn[:, :4], pbatch['ori_shape'],
        ratio_pad=pbatch['ratio_pad'])
    return predn
