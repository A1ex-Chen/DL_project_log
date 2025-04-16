@configurable
def __init__(self, **kwargs):
    """
        NOTE: this interface is experimental.
        """
    super().__init__(**kwargs)
    assert not self.mask_on and not self.keypoint_on, 'Mask/Keypoints not supported in Rotated ROIHeads.'
    assert not self.train_on_pred_boxes, 'train_on_pred_boxes not implemented for RROIHeads!'
