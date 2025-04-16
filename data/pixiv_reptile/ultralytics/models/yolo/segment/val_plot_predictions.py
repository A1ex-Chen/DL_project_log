def plot_predictions(self, batch, preds, ni):
    """Plots batch predictions with masks and bounding boxes."""
    plot_images(batch['img'], *output_to_target(preds[0], max_det=15), 
        torch.cat(self.plot_masks, dim=0) if len(self.plot_masks) else self
        .plot_masks, paths=batch['im_file'], fname=self.save_dir /
        f'val_batch{ni}_pred.jpg', names=self.names, on_plot=self.on_plot)
    self.plot_masks.clear()
