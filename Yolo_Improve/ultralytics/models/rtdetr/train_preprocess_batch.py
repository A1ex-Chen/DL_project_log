def preprocess_batch(self, batch):
    """
        Preprocess a batch of images. Scales and converts the images to float format.

        Args:
            batch (dict): Dictionary containing a batch of images, bboxes, and labels.

        Returns:
            (dict): Preprocessed batch.
        """
    batch = super().preprocess_batch(batch)
    bs = len(batch['img'])
    batch_idx = batch['batch_idx']
    gt_bbox, gt_class = [], []
    for i in range(bs):
        gt_bbox.append(batch['bboxes'][batch_idx == i].to(batch_idx.device))
        gt_class.append(batch['cls'][batch_idx == i].to(device=batch_idx.
            device, dtype=torch.long))
    return batch
