def get_validator(self):
    """Return an instance of OBBValidator for validation of YOLO model."""
    self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
    return yolo.obb.OBBValidator(self.test_loader, save_dir=self.save_dir,
        args=copy(self.args))
