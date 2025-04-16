def _reset(self):
    """Reset COCO API object."""
    if self.filename is None and not hasattr(self, 'coco_gt'):
        self.coco_gt = MaskCOCO()
