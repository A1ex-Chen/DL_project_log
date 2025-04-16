def get_validator(self):
    """Returns an instance of ClassificationValidator for validation."""
    self.loss_names = ['loss']
    return yolo.classify.ClassificationValidator(self.test_loader, self.
        save_dir, _callbacks=self.callbacks)
