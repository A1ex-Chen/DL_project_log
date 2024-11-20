def _prepare_pred(self, pred, pbatch):
    """Prepares and returns a batch with transformed bounding boxes and class labels."""
    predn = pred.clone()
    predn[..., [0, 2]] *= pbatch['ori_shape'][1] / self.args.imgsz
    predn[..., [1, 3]] *= pbatch['ori_shape'][0] / self.args.imgsz
    return predn.float()
