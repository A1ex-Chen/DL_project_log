def plot_predictions(self, batch, preds, ni):
    """Plots predicted bounding boxes on input images and saves the result."""
    plot_images(batch['img'], batch_idx=torch.arange(len(batch['img'])),
        cls=torch.argmax(preds, dim=1), fname=self.save_dir /
        f'val_batch{ni}_pred.jpg', names=self.names, on_plot=self.on_plot)
