def generate_detections_gpu(class_outputs, box_outputs, anchor_boxes,
    image_info, pre_nms_num_detections=1000, post_nms_num_detections=100,
    nms_threshold=0.3, class_agnostic_box=False, bbox_reg_weights=(10.0, 
    10.0, 5.0, 5.0)):
    """Generate the final detections given the model outputs (GPU version).

    Args:
    class_outputs: a tensor with shape [batch_size, N, num_classes], which
      stacks class logit outputs on all feature levels. The N is the number of
      total anchors on all levels. The num_classes is the number of classes
      predicted by the model. Note that the class_outputs here is the raw score.
    box_outputs: a tensor with shape [batch_size, N, num_classes*4], which
      stacks box regression outputs on all feature levels. The N is the number
      of total anchors on all levels.
    anchor_boxes: a tensor with shape [batch_size, N, 4], which stacks anchors
      on all feature levels. The N is the number of total anchors on all levels.
    image_info: a tensor of shape [batch_size, 5] which encodes each image's
      [height, width, scale, original_height, original_width].
    pre_nms_num_detections: an integer that specifies the number of candidates
      before NMS.
    post_nms_num_detections: an integer that specifies the number of candidates
      after NMS.
    nms_threshold: a float number to specify the IOU threshold of NMS.
    bbox_reg_weights: a list of 4 float scalars, which are default weights on
      (dx, dy, dw, dh) for normalizing bbox regression targets.

    Returns:
    a tuple of tensors corresponding to number of valid boxes,
    box coordinates, object categories for each boxes, and box scores stacked
    in batch_size.
    """
    with tf.name_scope('generate_detections'):
        batch_size, num_boxes, num_classes = class_outputs.get_shape().as_list(
            )
        softmax_class_outputs = tf.nn.softmax(class_outputs)
        scores = tf.slice(softmax_class_outputs, [0, 0, 1], [-1, -1, -1])
        box_classes = 2 if class_agnostic_box else num_classes
        boxes = tf.slice(tf.reshape(box_outputs, [batch_size, num_boxes,
            box_classes, 4]), [0, 0, 1, 0], [-1, -1, -1, -1])
        if class_agnostic_box:
            boxes = tf.tile(boxes, [1, 1, num_classes - 1, 1])
        anchor_boxes = tf.expand_dims(anchor_boxes, axis=2) * tf.ones([1, 1,
            num_classes - 1, 1])
        num_detections = num_boxes * (num_classes - 1)
        boxes = tf.reshape(boxes, [batch_size, num_detections, 4])
        scores = tf.reshape(scores, [batch_size, num_detections, 1])
        anchor_boxes = tf.reshape(anchor_boxes, [batch_size, num_detections, 4]
            )
        boxes = box_utils.decode_boxes(boxes, anchor_boxes, bbox_reg_weights)
        height = tf.expand_dims(image_info[:, 0:1], axis=-1)
        width = tf.expand_dims(image_info[:, 1:2], axis=-1)
        boxes = box_utils.clip_boxes(boxes, (height, width))
        pre_nms_boxes = box_utils.to_normalized_coordinates(boxes, height,
            width)
        pre_nms_boxes = tf.reshape(pre_nms_boxes, [batch_size, num_boxes, 
            num_classes - 1, 4])
        pre_nms_scores = tf.reshape(scores, [batch_size, num_boxes, 
            num_classes - 1])
        (post_nms_boxes, post_nms_scores, post_nms_classes,
            post_nms_num_valid_boxes) = (tf.image.
            combined_non_max_suppression(pre_nms_boxes, pre_nms_scores,
            max_output_size_per_class=pre_nms_num_detections,
            max_total_size=post_nms_num_detections, iou_threshold=
            nms_threshold, score_threshold=0.0, pad_per_class=False))
        post_nms_classes = post_nms_classes + 1
        post_nms_boxes = box_utils.to_absolute_coordinates(post_nms_boxes,
            height, width)
    return post_nms_num_valid_boxes, post_nms_boxes, tf.cast(post_nms_classes,
        dtype=tf.float32), post_nms_scores
