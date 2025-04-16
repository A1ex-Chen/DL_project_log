def preprocess(self, batch):
    """Preprocesses batch of images for YOLO training."""
    batch['img'] = batch['img'].to(self.device, non_blocking=True)
    batch['img'] = (batch['img'].half() if self.args.half else batch['img']
        .float()) / 255
    for k in ['batch_idx', 'cls', 'bboxes']:
        batch[k] = batch[k].to(self.device)
    if self.args.save_hybrid:
        height, width = batch['img'].shape[2:]
        nb = len(batch['img'])
        bboxes = batch['bboxes'] * torch.tensor((width, height, width,
            height), device=self.device)
        self.lb = [torch.cat([batch['cls'][batch['batch_idx'] == i], bboxes
            [batch['batch_idx'] == i]], dim=-1) for i in range(nb)
            ] if self.args.save_hybrid else []
    return batch
