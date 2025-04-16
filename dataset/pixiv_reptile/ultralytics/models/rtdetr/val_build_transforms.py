def build_transforms(self, hyp=None):
    """Temporary, only for evaluation."""
    if self.augment:
        hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
        hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
        transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
    else:
        transforms = Compose([])
    transforms.append(Format(bbox_format='xywh', normalize=True,
        return_mask=self.use_segments, return_keypoint=self.use_keypoints,
        batch_idx=True, mask_ratio=hyp.mask_ratio, mask_overlap=hyp.
        overlap_mask))
    return transforms
