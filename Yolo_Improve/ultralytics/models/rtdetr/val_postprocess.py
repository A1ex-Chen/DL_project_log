def postprocess(self, preds):
    """Apply Non-maximum suppression to prediction outputs."""
    if not isinstance(preds, (list, tuple)):
        preds = [preds, None]
    bs, _, nd = preds[0].shape
    bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
    bboxes *= self.args.imgsz
    outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
    for i, bbox in enumerate(bboxes):
        bbox = ops.xywh2xyxy(bbox)
        score, cls = scores[i].max(-1)
        pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)
        pred = pred[score.argsort(descending=True)]
        outputs[i] = pred
    return outputs
