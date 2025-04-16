def plot_training_samples(self, batch, ni):
    """Plots training samples with their annotations."""
    plot_images(images=batch['img'], batch_idx=batch['batch_idx'], cls=
        batch['cls'].squeeze(-1), bboxes=batch['bboxes'], paths=batch[
        'im_file'], fname=self.save_dir / f'train_batch{ni}.jpg', on_plot=
        self.on_plot)
