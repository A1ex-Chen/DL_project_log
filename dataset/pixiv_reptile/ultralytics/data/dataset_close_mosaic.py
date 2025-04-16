def close_mosaic(self, hyp):
    """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
    hyp.mosaic = 0.0
    hyp.copy_paste = 0.0
    hyp.mixup = 0.0
    self.transforms = self.build_transforms(hyp)
