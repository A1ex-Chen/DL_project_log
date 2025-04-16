def _prepare_batch(self, si, batch):
    """Prepares a batch for processing by converting keypoints to float and moving to device."""
    pbatch = super()._prepare_batch(si, batch)
    kpts = batch['keypoints'][batch['batch_idx'] == si]
    h, w = pbatch['imgsz']
    kpts = kpts.clone()
    kpts[..., 0] *= w
    kpts[..., 1] *= h
    kpts = ops.scale_coords(pbatch['imgsz'], kpts, pbatch['ori_shape'],
        ratio_pad=pbatch['ratio_pad'])
    pbatch['kpts'] = kpts
    return pbatch
