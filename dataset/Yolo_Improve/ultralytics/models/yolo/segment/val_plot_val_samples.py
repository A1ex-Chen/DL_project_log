def plot_val_samples(self, batch, ni):
    """Plots validation samples with bounding box labels."""
    plot_images(batch['img'], batch['batch_idx'], batch['cls'].squeeze(-1),
        batch['bboxes'], masks=batch['masks'], paths=batch['im_file'],
        fname=self.save_dir / f'val_batch{ni}_labels.jpg', names=self.names,
        on_plot=self.on_plot)
