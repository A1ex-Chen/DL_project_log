def get_validator(self):
    """Return an instance of SegmentationValidator for validation of YOLO model."""
    self.loss_names = 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss'
    return yolo.segment.SegmentationValidator(self.test_loader, save_dir=
        self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
