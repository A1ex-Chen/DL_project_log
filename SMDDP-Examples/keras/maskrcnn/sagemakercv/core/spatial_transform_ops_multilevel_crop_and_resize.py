def multilevel_crop_and_resize(features, boxes, output_size=7,
    is_gpu_inference=False):
    """Crop and resize on multilevel feature pyramid.

  Generate the (output_size, output_size) set of pixels for each input box
  by first locating the box into the correct feature level, and then cropping
  and resizing it using the correspoding feature map of that level.

  Args:
    features: A dictionary with key as pyramid level and value as features. The
      features are in shape of [batch_size, height_l, width_l, num_filters].
    boxes: A 3-D Tensor of shape [batch_size, num_boxes, 4]. Each row represents
      a box with [y1, x1, y2, x2] in un-normalized coordinates.
    output_size: A scalar to indicate the output crop size.
    is_gpu_inference: whether to build the model for GPU inference.

  Returns:
    A 5-D tensor representing feature crop of shape
    [batch_size, num_boxes, output_size, output_size, num_filters].
  """
    with tf.name_scope('multilevel_crop_and_resize'):
        levels = features.keys()
        min_level = min(levels)
        max_level = max(levels)
        _, max_feature_height, max_feature_width, _ = features[min_level
            ].get_shape().as_list()
        features_all = []
        for level in range(min_level, max_level + 1):
            features_all.append(tf.image.pad_to_bounding_box(features[level
                ], 0, 0, max_feature_height, max_feature_width))
        features_all = tf.stack(features_all, axis=1)
        box_width = tf.squeeze(boxes[:, :, 3:4] - boxes[:, :, 1:2], axis=-1)
        box_height = tf.squeeze(boxes[:, :, 2:3] - boxes[:, :, 0:1], axis=-1)
        areas_sqrt = tf.sqrt(box_height * box_width)
        levels = tf.math.floordiv(tf.math.log(tf.divide(areas_sqrt, 224.0)),
            tf.math.log(2.0)) + 4.0
        if not is_gpu_inference:
            levels = tf.cast(levels, dtype=tf.int32)
        levels = tf.minimum(float(max_level) if is_gpu_inference else
            max_level, tf.maximum(levels, float(min_level) if
            is_gpu_inference else min_level))
        scale_to_level = tf.cast(tf.pow(tf.constant(2.0), levels if
            is_gpu_inference else tf.cast(levels, tf.float32)), dtype=boxes
            .dtype)
        boxes /= tf.expand_dims(scale_to_level, axis=2)
        box_width /= scale_to_level
        box_height /= scale_to_level
        boxes = tf.concat([boxes[:, :, 0:2], tf.expand_dims(box_height, -1),
            tf.expand_dims(box_width, -1)], axis=-1)
        levels -= min_level
        level_strides = tf.pow([[2.0]], levels if is_gpu_inference else tf.
            cast(levels, tf.float32))
        boundary = tf.cast(tf.concat([tf.expand_dims([[tf.cast(
            max_feature_height, tf.float32)]] / level_strides - 1, axis=-1),
            tf.expand_dims([[tf.cast(max_feature_width, tf.float32)]] /
            level_strides - 1, axis=-1)], axis=-1), boxes.dtype)
    return selective_crop_and_resize(features_all, boxes, levels, boundary,
        output_size, is_gpu_inference)
