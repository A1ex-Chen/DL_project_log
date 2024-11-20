def plot_training_samples(self, batch, ni):
    """Plots training samples with their annotations."""
    plot_images(images=batch['img'], batch_idx=torch.arange(len(batch['img'
        ])), cls=batch['cls'].view(-1), fname=self.save_dir /
        f'train_batch{ni}.jpg', on_plot=self.on_plot)
