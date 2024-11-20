def selective_crop_and_resize(features, boxes, box_levels, boundaries,
    output_size=7, is_gpu_inference=False):
    """Crop and resize boxes on a set of feature maps.

  Given multiple features maps indexed by different levels, and a set of boxes
  where each box is mapped to a certain level, it selectively crops and resizes
  boxes from the corresponding feature maps to generate the box features.

  We follow the ROIAlign technique (see https://arxiv.org/pdf/1703.06870.pdf,
  figure 3 for reference). Specifically, for each feature map, we select an
  (output_size, output_size) set of pixels corresponding to the box location,
  and then use bilinear interpolation to select the feature value for each
  pixel.

  For performance, we perform the gather and interpolation on all layers as a
  single operation. This is op the multi-level features are first stacked and
  gathered into [2*output_size, 2*output_size] feature points. Then bilinear
  interpolation is performed on the gathered feature points to generate
  [output_size, output_size] RoIAlign feature map.

  Here is the step-by-step algorithm:
    1. The multi-level features are gathered into a
       [batch_size, num_boxes, output_size*2, output_size*2, num_filters]
       Tensor. The Tensor contains four neighboring feature points for each
       vertice in the output grid.
    2. Compute the interpolation kernel of shape
       [batch_size, num_boxes, output_size*2, output_size*2]. The last 2 axis
       can be seen as stacking 2x2 interpolation kernels for all vertices in the
       output grid.
    3. Element-wise multiply the gathered features and interpolation kernel.
       Then apply 2x2 average pooling to reduce spatial dimension to
       output_size.

  Args:
    features: a 5-D tensor of shape
      [batch_size, num_levels, max_height, max_width, num_filters] where
      cropping and resizing are based.
    boxes: a 3-D tensor of shape [batch_size, num_boxes, 4] encoding the
      information of each box w.r.t. the corresponding feature map.
      boxes[:, :, 0:2] are the grid position in (y, x) (float) of the top-left
      corner of each box. boxes[:, :, 2:4] are the box sizes in (h, w) (float)
        in terms of the number of pixels of the corresponding feature map size.
    box_levels: a 3-D tensor of shape [batch_size, num_boxes, 1] representing
      the 0-based corresponding feature level index of each box.
    boundaries: a 3-D tensor of shape [batch_size, num_boxes, 2] representing
      the boundary (in (y, x)) of the corresponding feature map for each box.
      Any resampled grid points that go beyond the bounary will be clipped.
    output_size: a scalar indicating the output crop size.
    is_gpu_inference: whether to build the model for GPU inference.

  Returns:
    features_per_box: a 5-D tensor of shape
      [batch_size, num_boxes, output_size, output_size, num_filters]
      representing the cropped features.
  """
    (batch_size, num_levels, max_feature_height, max_feature_width, num_filters
        ) = features.get_shape().as_list()
    _, num_boxes, _ = boxes.get_shape().as_list()
    box_grid_x = []
    box_grid_y = []
    for i in range(output_size):
        box_grid_x.append(boxes[:, :, 1:2] + (i + 0.5) * boxes[:, :, 3:4] /
            output_size)
        box_grid_y.append(boxes[:, :, 0:1] + (i + 0.5) * boxes[:, :, 2:3] /
            output_size)
    box_grid_x = tf.concat(box_grid_x, axis=-1)
    box_grid_y = tf.concat(box_grid_y, axis=-1)
    box_grid_y0 = tf.floor(box_grid_y)
    box_grid_x0 = tf.floor(box_grid_x)
    box_grid_x0 = tf.maximum(0.0, box_grid_x0)
    box_grid_y0 = tf.maximum(0.0, box_grid_y0)
    box_gridx0x1 = tf.stack([tf.minimum(box_grid_x0, boundaries[:, :, 1:2]),
        tf.minimum(box_grid_x0 + 1, boundaries[:, :, 1:2])], axis=3)
    box_gridy0y1 = tf.stack([tf.minimum(box_grid_y0, boundaries[:, :, 0:1]),
        tf.minimum(box_grid_y0 + 1, boundaries[:, :, 0:1])], axis=3)
    x_indices = tf.reshape(box_gridx0x1, [batch_size, num_boxes, 
        output_size * 2])
    y_indices = tf.reshape(box_gridy0y1, [batch_size, num_boxes, 
        output_size * 2])
    indices_dtype = tf.float32 if is_gpu_inference else tf.int32
    if not is_gpu_inference:
        x_indices = tf.cast(x_indices, tf.int32)
        y_indices = tf.cast(y_indices, tf.int32)
    height_dim_offset = max_feature_width
    level_dim_offset = max_feature_height * height_dim_offset
    batch_dim_offset = num_levels * level_dim_offset
    batch_dim_indices = tf.reshape(tf.range(batch_size, dtype=indices_dtype
        ) * batch_dim_offset, [batch_size, 1, 1, 1]) * tf.ones([1,
        num_boxes, output_size * 2, output_size * 2], dtype=indices_dtype)
    box_level_indices = tf.reshape(box_levels * level_dim_offset, [
        batch_size, num_boxes, 1, 1]) * tf.ones([1, 1, output_size * 2, 
        output_size * 2], dtype=indices_dtype)
    height_indices = tf.reshape(y_indices * height_dim_offset, [batch_size,
        num_boxes, output_size * 2, 1]) * tf.ones([1, 1, 1, output_size * 2
        ], dtype=indices_dtype)
    width_indices = tf.reshape(x_indices, [batch_size, num_boxes, 1, 
        output_size * 2]) * tf.ones([1, 1, output_size * 2, 1], dtype=
        indices_dtype)
    if True:
        batch_dim_indices = tf.cast(batch_dim_indices, tf.float32)
        box_level_indices = tf.cast(box_level_indices, tf.float32)
        height_indices = tf.cast(height_indices, tf.float32)
        width_indices = tf.cast(width_indices, tf.float32)
        indices = tf.add_n([batch_dim_indices, box_level_indices,
            height_indices, width_indices])
        indices = tf.cast(indices, tf.int32)
    else:
        indices = tf.add_n([batch_dim_indices, box_level_indices,
            height_indices, width_indices])
    if batch_size == 1:
        indices = tf.reshape(indices, [1, -1])
        if is_gpu_inference:
            indices = tf.cast(indices, dtype=tf.int32)
        features = tf.reshape(features, [1, -1, num_filters])
        features_per_box = tf.gather(features, indices, axis=1)
    else:
        indices = tf.reshape(indices, [-1])
        if is_gpu_inference:
            indices = tf.cast(indices, dtype=tf.int32)
        features = tf.reshape(features, [-1, num_filters])
        features_per_box = tf.gather(features, indices)
    features_per_box = tf.reshape(features_per_box, [batch_size, num_boxes,
        output_size * 2, output_size * 2, num_filters])
    ly = box_grid_y - box_grid_y0
    lx = box_grid_x - box_grid_x0
    hy = 1.0 - ly
    hx = 1.0 - lx
    kernel_x = tf.reshape(tf.stack([hx, lx], axis=3), [batch_size,
        num_boxes, 1, output_size * 2])
    kernel_y = tf.reshape(tf.stack([hy, ly], axis=3), [batch_size,
        num_boxes, output_size * 2, 1])
    interpolation_kernel = kernel_y * kernel_x * 4
    features_per_box *= tf.cast(tf.expand_dims(interpolation_kernel, axis=4
        ), dtype=features_per_box.dtype)
    features_per_box = tf.reshape(features_per_box, [batch_size * num_boxes,
        output_size * 2, output_size * 2, num_filters])
    features_per_box = tf.nn.avg_pool2d(features_per_box, ksize=[1, 2, 2, 1
        ], strides=[1, 2, 2, 1], padding='VALID')
    features_per_box = tf.reshape(features_per_box, [batch_size, num_boxes,
        output_size, output_size, num_filters])
    return features_per_box
