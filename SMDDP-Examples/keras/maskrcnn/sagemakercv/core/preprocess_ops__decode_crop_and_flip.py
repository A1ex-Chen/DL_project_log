def _decode_crop_and_flip(image_buffer, bbox, num_channels):
    """Crops the given image to a random part of the image, and randomly flips.
    We use the fused decode_and_crop op, which performs better than the two ops
    used separately in series, but note that this requires that the image be
    passed in as an un-decoded string Tensor.
    Args:
    image_buffer: scalar string Tensor representing the raw JPEG image buffer.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    num_channels: Integer depth of the image buffer for decoding.
    Returns:
    3-D tensor with cropped image.
    """
    sample_distorted_bounding_box = tf.raw_ops.SampleDistortedBoundingBoxV2(
        image_size=tf.image.extract_jpeg_shape(image_buffer),
        bounding_boxes=bbox, min_object_covered=0.1, aspect_ratio_range=[
        0.75, 1.33], area_range=[0.05, 1.0], max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    cropped = tf.image.decode_and_crop_jpeg(image_buffer, crop_window,
        channels=num_channels)
    cropped = tf.image.random_flip_left_right(cropped)
    return cropped
