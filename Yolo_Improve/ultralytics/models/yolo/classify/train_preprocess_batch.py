def preprocess_batch(self, batch):
    """Preprocesses a batch of images and classes."""
    batch['img'] = batch['img'].to(self.device)
    batch['cls'] = batch['cls'].to(self.device)
    return batch
