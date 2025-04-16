def plot_predictions(self, batch, preds, ni):
    """Plots predictions for YOLO model."""
    pred_kpts = torch.cat([p[:, 6:].view(-1, *self.kpt_shape) for p in
        preds], 0)
    plot_images(batch['img'], *output_to_target(preds, max_det=self.args.
        max_det), kpts=pred_kpts, paths=batch['im_file'], fname=self.
        save_dir / f'val_batch{ni}_pred.jpg', names=self.names, on_plot=
        self.on_plot)
