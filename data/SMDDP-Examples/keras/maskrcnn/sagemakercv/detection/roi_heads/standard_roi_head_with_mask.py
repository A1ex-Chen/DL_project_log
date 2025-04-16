@property
def with_mask(self):
    """bool: whether the RoI head contains a `mask_head`"""
    return hasattr(self, 'mask_head') and self.mask_head is not None
