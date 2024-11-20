def plot_val_samples(self, batch, ni):
    """Plots and saves validation set samples with predicted bounding boxes and keypoints."""
    plot_images(batch['img'], batch['batch_idx'], batch['cls'].squeeze(-1),
        batch['bboxes'], kpts=batch['keypoints'], paths=batch['im_file'],
        fname=self.save_dir / f'val_batch{ni}_labels.jpg', names=self.names,
        on_plot=self.on_plot)
