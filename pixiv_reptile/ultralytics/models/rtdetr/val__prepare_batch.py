def _prepare_batch(self, si, batch):
    """Prepares a batch for training or inference by applying transformations."""
    idx = batch['batch_idx'] == si
    cls = batch['cls'][idx].squeeze(-1)
    bbox = batch['bboxes'][idx]
    ori_shape = batch['ori_shape'][si]
    imgsz = batch['img'].shape[2:]
    ratio_pad = batch['ratio_pad'][si]
    if len(cls):
        bbox = ops.xywh2xyxy(bbox)
        bbox[..., [0, 2]] *= ori_shape[1]
        bbox[..., [1, 3]] *= ori_shape[0]
    return {'cls': cls, 'bbox': bbox, 'ori_shape': ori_shape, 'imgsz':
        imgsz, 'ratio_pad': ratio_pad}
