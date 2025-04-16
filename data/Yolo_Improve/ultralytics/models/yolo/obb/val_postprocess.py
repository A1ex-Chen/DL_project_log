def postprocess(self, preds):
    """Apply Non-maximum suppression to prediction outputs."""
    return ops.non_max_suppression(preds, self.args.conf, self.args.iou,
        labels=self.lb, nc=self.nc, multi_label=True, agnostic=self.args.
        single_cls, max_det=self.args.max_det, rotated=True)
