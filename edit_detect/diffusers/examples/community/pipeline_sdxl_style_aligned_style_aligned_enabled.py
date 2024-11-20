@property
def style_aligned_enabled(self):
    """Returns whether StyleAligned has been enabled in the pipeline or not."""
    return hasattr(self, '_style_aligned_norm_layers'
        ) and self._style_aligned_norm_layers is not None
