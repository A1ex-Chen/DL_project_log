def _format_segments(self, instances, cls, w, h):
    """Convert polygon points to bitmap."""
    segments = instances.segments
    if self.mask_overlap:
        masks, sorted_idx = polygons2masks_overlap((h, w), segments,
            downsample_ratio=self.mask_ratio)
        masks = masks[None]
        instances = instances[sorted_idx]
        cls = cls[sorted_idx]
    else:
        masks = polygons2masks((h, w), segments, color=1, downsample_ratio=
            self.mask_ratio)
    return masks, instances, cls
