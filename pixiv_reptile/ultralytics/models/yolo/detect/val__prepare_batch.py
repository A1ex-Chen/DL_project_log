def _prepare_batch(self, si, batch):
    """Prepares a batch of images and annotations for validation."""
    idx = batch['batch_idx'] == si
    cls = batch['cls'][idx].squeeze(-1)
    bbox = batch['bboxes'][idx]
    ori_shape = batch['ori_shape'][si]
    imgsz = batch['img'].shape[2:]
    ratio_pad = batch['ratio_pad'][si]
    if len(cls):
        bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[
            1, 0, 1, 0]]
        ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)
    return {'cls': cls, 'bbox': bbox, 'ori_shape': ori_shape, 'imgsz':
        imgsz, 'ratio_pad': ratio_pad}
