def _denormalize_to_coco_bbox(bbox, height, width):
    """Denormalize bounding box.

  Args:
    bbox: numpy.array[float]. Normalized bounding box. Format: ['ymin', 'xmin',
      'ymax', 'xmax'].
    height: int. image height.
    width: int. image width.

  Returns:
    [x, y, width, height]
  """
    y1, x1, y2, x2 = bbox
    y1 *= height
    x1 *= width
    y2 *= height
    x2 *= width
    box_height = y2 - y1
    box_width = x2 - x1
    return [float(x1), float(y1), float(box_width), float(box_height)]
