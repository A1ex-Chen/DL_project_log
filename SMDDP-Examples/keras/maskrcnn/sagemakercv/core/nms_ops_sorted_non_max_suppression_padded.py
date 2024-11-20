def sorted_non_max_suppression_padded(scores, boxes, max_output_size,
    iou_threshold):
    """A wrapper that handles non-maximum suppression.

    Assumption:
        * The boxes are sorted by scores unless the box is a dot (all coordinates
            are zero).
        * Boxes with higher scores can be used to suppress boxes with lower scores.

    The overal design of the algorithm is to handle boxes tile-by-tile:

    boxes = boxes.pad_to_multiply_of(tile_size)
    num_tiles = len(boxes) // tile_size
    output_boxes = []
    for i in range(num_tiles):
        box_tile = boxes[i*tile_size : (i+1)*tile_size]
        for j in range(i - 1):
            suppressing_tile = boxes[j*tile_size : (j+1)*tile_size]
            iou = bbox_overlap(box_tile, suppressing_tile)
            # if the box is suppressed in iou, clear it to a dot
            box_tile *= _update_boxes(iou)
        # Iteratively handle the diagnal tile.
        iou = _box_overlap(box_tile, box_tile)
        iou_changed = True
        while iou_changed:
            # boxes that are not suppressed by anything else
            suppressing_boxes = _get_suppressing_boxes(iou)
            # boxes that are suppressed by suppressing_boxes
            suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)
            # clear iou to 0 for boxes that are suppressed, as they cannot be used
            # to suppress other boxes any more
            new_iou = _clear_iou(iou, suppressed_boxes)
            iou_changed = (new_iou != iou)
            iou = new_iou
        # remaining boxes that can still suppress others, are selected boxes.
        output_boxes.append(_get_suppressing_boxes(iou))
        if len(output_boxes) >= max_output_size:
            break

    Args:
        scores: a tensor with a shape of [batch_size, anchors].
        boxes: a tensor with a shape of [batch_size, anchors, 4].
        max_output_size: a scalar integer `Tensor` representing the maximum number
            of boxes to be selected by non max suppression.
        iou_threshold: a float representing the threshold for deciding whether boxes
            overlap too much with respect to IOU.

    Returns:
        nms_scores: a tensor with a shape of [batch_size, anchors]. It has same
            dtype as input scores.
        nms_proposals: a tensor with a shape of [batch_size, anchors, 4]. It has
            same dtype as input boxes.
    """
    batch_size = tf.shape(boxes)[0]
    num_boxes = tf.shape(boxes)[1]
    pad = tf.cast(tf.ceil(tf.cast(num_boxes, tf.float32) / NMS_TILE_SIZE),
        tf.int32) * NMS_TILE_SIZE - num_boxes
    boxes = tf.pad(tf.cast(boxes, tf.float32), [[0, 0], [0, pad], [0, 0]])
    scores = tf.pad(tf.cast(scores, tf.float32), [[0, 0], [0, pad]])
    num_boxes += pad

    def _loop_cond(unused_boxes, unused_threshold, output_size, idx):
        return tf.logical_and(tf.reduce_min(output_size) < max_output_size,
            idx < num_boxes // NMS_TILE_SIZE)
    selected_boxes, _, output_size, _ = tf.while_loop(_loop_cond,
        _suppression_loop_body, [boxes, iou_threshold, tf.zeros([batch_size
        ], tf.int32), tf.constant(0)])
    idx = num_boxes - tf.cast(tf.nn.top_k(tf.cast(tf.reduce_any(
        selected_boxes > 0, [2]), tf.int32) * tf.expand_dims(tf.range(
        num_boxes, 0, -1), 0), max_output_size)[0], tf.int32)
    idx = tf.minimum(idx, num_boxes - 1)
    idx = tf.reshape(idx + tf.reshape(tf.range(batch_size) * num_boxes, [-1,
        1]), [-1])
    boxes = tf.reshape(tf.gather(tf.reshape(boxes, [-1, 4]), idx), [
        batch_size, max_output_size, 4])
    boxes = boxes * tf.cast(tf.reshape(tf.range(max_output_size), [1, -1, 1
        ]) < tf.reshape(output_size, [-1, 1, 1]), boxes.dtype)
    scores = tf.reshape(tf.gather(tf.reshape(scores, [-1, 1]), idx), [
        batch_size, max_output_size])
    scores = scores * tf.cast(tf.reshape(tf.range(max_output_size), [1, -1]
        ) < tf.reshape(output_size, [-1, 1]), scores.dtype)
    return scores, boxes
