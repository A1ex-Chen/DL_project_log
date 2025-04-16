def plot_training_samples(self, batch, ni):
    """Plot a batch of training samples with annotated class labels, bounding boxes, and keypoints."""
    images = batch['img']
    kpts = batch['keypoints']
    cls = batch['cls'].squeeze(-1)
    bboxes = batch['bboxes']
    paths = batch['im_file']
    batch_idx = batch['batch_idx']
    plot_images(images, batch_idx, cls, bboxes, kpts=kpts, paths=paths,
        fname=self.save_dir / f'train_batch{ni}.jpg', on_plot=self.on_plot)
