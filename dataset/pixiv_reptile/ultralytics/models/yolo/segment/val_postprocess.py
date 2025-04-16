def postprocess(self, preds):
    """Post-processes YOLO predictions and returns output detections with proto."""
    p = ops.non_max_suppression(preds[0], self.args.conf, self.args.iou,
        labels=self.lb, multi_label=True, agnostic=self.args.single_cls,
        max_det=self.args.max_det, nc=self.nc)
    proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]
    return p, proto
