@torch.jit.unused
def to_bitmasks(self, boxes: torch.Tensor, height, width, threshold=0.5):
    """
        Args: see documentation of :func:`paste_masks_in_image`.
        """
    from detectron2.layers.mask_ops import paste_masks_in_image, _paste_masks_tensor_shape
    if torch.jit.is_tracing():
        if isinstance(height, torch.Tensor):
            paste_func = _paste_masks_tensor_shape
        else:
            paste_func = paste_masks_in_image
    else:
        paste_func = retry_if_cuda_oom(paste_masks_in_image)
    bitmasks = paste_func(self.tensor, boxes.tensor, (height, width),
        threshold=threshold)
    return BitMasks(bitmasks)
