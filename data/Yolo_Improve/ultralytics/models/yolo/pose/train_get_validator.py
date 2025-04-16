def get_validator(self):
    """Returns an instance of the PoseValidator class for validation."""
    self.loss_names = ('box_loss', 'pose_loss', 'kobj_loss', 'cls_loss',
        'dfl_loss')
    return yolo.pose.PoseValidator(self.test_loader, save_dir=self.save_dir,
        args=copy(self.args), _callbacks=self.callbacks)
