def preprocess(self, batch):
    """Preprocesses batch by converting masks to float and sending to device."""
    batch = super().preprocess(batch)
    batch['masks'] = batch['masks'].to(self.device).float()
    return batch
