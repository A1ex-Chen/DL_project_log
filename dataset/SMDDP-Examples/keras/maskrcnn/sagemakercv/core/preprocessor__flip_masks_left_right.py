def _flip_masks_left_right(masks):
    """Left-right flip masks.

  Args:
    masks: rank 3 float32 tensor with shape
      [num_instances, height, width] representing instance masks.

  Returns:
    flipped masks: rank 3 float32 tensor with shape
      [num_instances, height, width] representing instance masks.
  """
    return masks[:, :, ::-1]
