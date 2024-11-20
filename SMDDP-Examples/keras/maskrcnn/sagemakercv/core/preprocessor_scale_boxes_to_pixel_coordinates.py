def scale_boxes_to_pixel_coordinates(image, boxes, keypoints=None):
    """Scales boxes from normalized to pixel coordinates.

  Args:
    image: A 3D float32 tensor of shape [height, width, channels].
    boxes: A 2D float32 tensor of shape [num_boxes, 4] containing the bounding
      boxes in normalized coordinates. Each row is of the form
      [ymin, xmin, ymax, xmax].
    keypoints: (optional) rank 3 float32 tensor with shape
      [num_instances, num_keypoints, 2]. The keypoints are in y-x normalized
      coordinates.

  Returns:
    image: unchanged input image.
    scaled_boxes: a 2D float32 tensor of shape [num_boxes, 4] containing the
      bounding boxes in pixel coordinates.
    scaled_keypoints: a 3D float32 tensor with shape
      [num_instances, num_keypoints, 2] containing the keypoints in pixel
      coordinates.
  """
    boxlist = box_list.BoxList(boxes)
    image_height = tf.shape(input=image)[0]
    image_width = tf.shape(input=image)[1]
    scaled_boxes = box_list_scale(boxlist, image_height, image_width).get()
    result = [image, scaled_boxes]
    if keypoints is not None:
        scaled_keypoints = keypoint_scale(keypoints, image_height, image_width)
        result.append(scaled_keypoints)
    return tuple(result)
