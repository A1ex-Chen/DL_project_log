def plot_val_samples(self, batch, ni):
    """Plot validation image samples."""
    plot_images(images=batch['img'], batch_idx=torch.arange(len(batch['img'
        ])), cls=batch['cls'].view(-1), fname=self.save_dir /
        f'val_batch{ni}_labels.jpg', names=self.names, on_plot=self.on_plot)
