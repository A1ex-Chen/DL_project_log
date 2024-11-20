def postprocess(self, preds):
    """Apply non-maximum suppression and return detections with high confidence scores."""
    return ops.non_max_suppression(preds, self.args.conf, self.args.iou,
        labels=self.lb, multi_label=True, agnostic=self.args.single_cls,
        max_det=self.args.max_det, nc=self.nc)
