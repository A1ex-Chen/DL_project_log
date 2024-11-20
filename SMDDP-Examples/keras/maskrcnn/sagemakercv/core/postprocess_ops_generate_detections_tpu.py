def generate_detections_tpu(class_outputs, box_outputs, anchor_boxes,
    image_info, pre_nms_num_detections=1000, post_nms_num_detections=100,
    nms_threshold=0.3, bbox_reg_weights=(10.0, 10.0, 5.0, 5.0)):
    """Generate the final detections given the model outputs (TPU version).

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
        batch_size, _, _ = class_outputs.get_shape().as_list()
        softmax_class_outputs = tf.nn.softmax(class_outputs)
        num_valid_boxes, box_coordinates, box_classes, box_scores = [], [], [
            ], []
        for i in range(batch_size):
            result = generate_detections_per_image_tpu(softmax_class_outputs
                [i], box_outputs[i], anchor_boxes[i], image_info[i],
                pre_nms_num_detections, post_nms_num_detections,
                nms_threshold, bbox_reg_weights)
            num_valid_boxes.append(result[0])
            box_coordinates.append(result[1])
            box_classes.append(result[2])
            box_scores.append(result[3])
        num_valid_boxes = tf.stack(num_valid_boxes)
        box_coordinates = tf.stack(box_coordinates)
        box_classes = tf.stack(box_classes)
        box_scores = tf.stack(box_scores)
    return num_valid_boxes, box_coordinates, box_classes, box_scores
