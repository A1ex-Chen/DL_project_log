def plot_training_samples(self, batch, ni):
    """Creates a plot of training sample images with labels and box coordinates."""
    plot_images(batch['img'], batch['batch_idx'], batch['cls'].squeeze(-1),
        batch['bboxes'], masks=batch['masks'], paths=batch['im_file'],
        fname=self.save_dir / f'train_batch{ni}.jpg', on_plot=self.on_plot)
