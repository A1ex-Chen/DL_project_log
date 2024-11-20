def on_val_image_end(self, pred, predn, path, names, im):
    if self.wandb:
        self.wandb.val_one_image(pred, predn, path, names, im)
    if self.clearml:
        self.clearml.log_image_with_boxes(path, pred, names, im)
