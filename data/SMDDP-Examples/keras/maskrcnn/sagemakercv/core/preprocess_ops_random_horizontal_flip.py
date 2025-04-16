def random_horizontal_flip(image, boxes=None, masks=None, seed=None):
    """Random horizontal flip the image, boxes, and masks.

    Args:
    image: a tensor of shape [height, width, 3] representing the image.
    boxes: (Optional) a tensor of shape [num_boxes, 4] represneting the box
      corners in normalized coordinates.
    masks: (Optional) a tensor of shape [num_masks, height, width]
      representing the object masks. Note that the size of the mask is the
      same as the image.

    Returns:
    image: the processed image tensor after being randomly flipped.
    boxes: None or the processed box tensor after being randomly flipped.
    masks: None or the processed mask tensor after being randomly flipped.
    """
    return preprocessor.random_horizontal_flip(image, boxes, masks, seed=seed)
