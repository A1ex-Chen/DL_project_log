def get_validator(self):
    """Returns a DetectionValidator for YOLO model validation."""
    self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
    return yolo.detect.DetectionValidator(self.test_loader, save_dir=self.
        save_dir, args=copy(self.args), _callbacks=self.callbacks)
