def _prepare_pred(self, pred, pbatch):
    """Prepares and returns a batch for OBB validation with scaled and padded bounding boxes."""
    predn = pred.clone()
    ops.scale_boxes(pbatch['imgsz'], predn[:, :4], pbatch['ori_shape'],
        ratio_pad=pbatch['ratio_pad'], xywh=True)
    return predn
