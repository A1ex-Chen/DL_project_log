@torch.jit.script_if_tracing
def _paste_masks_tensor_shape(masks: torch.Tensor, boxes: torch.Tensor,
    image_shape: Tuple[torch.Tensor, torch.Tensor], threshold: float=0.5):
    """
    A wrapper of paste_masks_in_image where image_shape is Tensor.
    During tracing, shapes might be tensors instead of ints. The Tensor->int
    conversion should be scripted rather than traced.
    """
    return paste_masks_in_image(masks, boxes, (int(image_shape[0]), int(
        image_shape[1])), threshold)
