def decode_crop_and_flip(image_bytes: tf.Tensor) ->tf.Tensor:
    """Crops an image to a random part of the image, then randomly flips.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.

  Returns:
    A decoded and cropped image `Tensor`.

  """
    decoded = image_bytes.dtype != tf.string
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    shape = tf.shape(image_bytes) if decoded else tf.image.extract_jpeg_shape(
        image_bytes)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape, bounding_boxes=bbox, min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33], area_range=[0.05, 1.0],
        max_attempts=100, use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box
    offset_height, offset_width, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_height, offset_width, target_height,
        target_width])
    if decoded:
        cropped = tf.image.crop_to_bounding_box(image_bytes, offset_height=
            offset_height, offset_width=offset_width, target_height=
            target_height, target_width=target_width)
    else:
        cropped = tf.image.decode_and_crop_jpeg(image_bytes, crop_window,
            channels=3)
    cropped = tf.image.random_flip_left_right(cropped)
    return cropped
