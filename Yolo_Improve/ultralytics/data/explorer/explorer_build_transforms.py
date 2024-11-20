def build_transforms(self, hyp: IterableSimpleNamespace=None):
    """Creates transforms for dataset images without resizing."""
    return Format(bbox_format='xyxy', normalize=False, return_mask=self.
        use_segments, return_keypoint=self.use_keypoints, batch_idx=True,
        mask_ratio=hyp.mask_ratio, mask_overlap=hyp.overlap_mask)
