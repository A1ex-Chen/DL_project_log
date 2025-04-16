def plot_predictions(self, batch, preds, ni):
    """Plots predicted bounding boxes on input images and saves the result."""
    plot_images(batch['img'], *output_to_target(preds, max_det=self.args.
        max_det), paths=batch['im_file'], fname=self.save_dir /
        f'val_batch{ni}_pred.jpg', names=self.names, on_plot=self.on_plot)
