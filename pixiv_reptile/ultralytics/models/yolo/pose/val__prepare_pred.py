def _prepare_pred(self, pred, pbatch):
    """Prepares and scales keypoints in a batch for pose processing."""
    predn = super()._prepare_pred(pred, pbatch)
    nk = pbatch['kpts'].shape[1]
    pred_kpts = predn[:, 6:].view(len(predn), nk, -1)
    ops.scale_coords(pbatch['imgsz'], pred_kpts, pbatch['ori_shape'],
        ratio_pad=pbatch['ratio_pad'])
    return predn, pred_kpts
