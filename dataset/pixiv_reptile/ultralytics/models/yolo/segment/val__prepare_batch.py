def _prepare_batch(self, si, batch):
    """Prepares a batch for training or inference by processing images and targets."""
    prepared_batch = super()._prepare_batch(si, batch)
    midx = [si] if self.args.overlap_mask else batch['batch_idx'] == si
    prepared_batch['masks'] = batch['masks'][midx]
    return prepared_batch
