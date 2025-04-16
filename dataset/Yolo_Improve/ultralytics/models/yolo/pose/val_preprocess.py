def preprocess(self, batch):
    """Preprocesses the batch by converting the 'keypoints' data into a float and moving it to the device."""
    batch = super().preprocess(batch)
    batch['keypoints'] = batch['keypoints'].to(self.device).float()
    return batch
