def _prepare_batch(self, si, batch):
    """Prepares and returns a batch for OBB validation."""
    idx = batch['batch_idx'] == si
    cls = batch['cls'][idx].squeeze(-1)
    bbox = batch['bboxes'][idx]
    ori_shape = batch['ori_shape'][si]
    imgsz = batch['img'].shape[2:]
    ratio_pad = batch['ratio_pad'][si]
    if len(cls):
        bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1,
            0]])
        ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad, xywh=True)
    return {'cls': cls, 'bbox': bbox, 'ori_shape': ori_shape, 'imgsz':
        imgsz, 'ratio_pad': ratio_pad}
