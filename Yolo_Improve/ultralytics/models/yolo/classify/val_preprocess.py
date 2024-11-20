def preprocess(self, batch):
    """Preprocesses input batch and returns it."""
    batch['img'] = batch['img'].to(self.device, non_blocking=True)
    batch['img'] = batch['img'].half() if self.args.half else batch['img'
        ].float()
    batch['cls'] = batch['cls'].to(self.device)
    return batch
