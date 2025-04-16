@torch.jit.script_if_tracing
def paste_masks_in_image(masks: torch.Tensor, boxes: torch.Tensor,
    image_shape: Tuple[int, int], threshold: float=0.5):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.

    Note:
        This is a complicated but more accurate implementation. In actual deployment, it is
        often enough to use a faster but less accurate implementation.
        See :func:`paste_mask_in_image_old` in this file for an alternative implementation.

    Args:
        masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
            detected object instances in the image and Hmask, Wmask are the mask width and mask
            height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
        boxes (Boxes or Tensor): A Boxes of length Bimg or Tensor of shape (Bimg, 4).
            boxes[i] and masks[i] correspond to the same object instance.
        image_shape (tuple): height, width
        threshold (float): A threshold in [0, 1] for converting the (soft) masks to
            binary masks.

    Returns:
        img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
        number of detected object instances and Himage, Wimage are the image width
        and height. img_masks[i] is a binary mask for object instance i.
    """
    assert masks.shape[-1] == masks.shape[-2
        ], 'Only square mask predictions are supported'
    N = len(masks)
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape
    img_h, img_w = image_shape
    if device.type == 'cpu' or torch.jit.is_scripting():
        num_chunks = N
    else:
        num_chunks = int(np.ceil(N * int(img_h) * int(img_w) *
            BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert num_chunks <= N, 'Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it'
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)
    img_masks = torch.zeros(N, img_h, img_w, device=device, dtype=torch.
        bool if threshold >= 0 else torch.uint8)
    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(masks[inds, None, :, :],
            boxes[inds], img_h, img_w, skip_empty=device.type == 'cpu')
        if threshold >= 0:
            masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
        else:
            masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)
        if torch.jit.is_scripting():
            img_masks[inds] = masks_chunk
        else:
            img_masks[(inds,) + spatial_inds] = masks_chunk
    return img_masks
