def _suppression_loop_body(boxes, iou_threshold, output_size, idx):
    """Process boxes in the range [idx*NMS_TILE_SIZE, (idx+1)*NMS_TILE_SIZE).
    Args:
      boxes: a tensor with a shape of [batch_size, anchors, 4].
      iou_threshold: a float representing the threshold for deciding whether boxes
        overlap too much with respect to IOU.
      output_size: an int32 tensor of size [batch_size]. Representing the number
        of selected boxes for each batch.
      idx: an integer scalar representing induction variable.
    Returns:
      boxes: updated boxes.
      iou_threshold: pass down iou_threshold to the next iteration.
      output_size: the updated output_size.
      idx: the updated induction variable.
    """
    num_tiles = tf.shape(input=boxes)[1] // NMS_TILE_SIZE
    batch_size = tf.shape(input=boxes)[0]
    box_slice = tf.slice(boxes, [0, idx * NMS_TILE_SIZE, 0], [batch_size,
        NMS_TILE_SIZE, 4])
    _, box_slice, _, _ = tf.while_loop(cond=lambda _boxes, _box_slice,
        _threshold, inner_idx: inner_idx < idx, body=_cross_suppression,
        loop_vars=[boxes, box_slice, iou_threshold, tf.constant(0)])
    iou = bbox_overlap(box_slice, box_slice)
    mask = tf.expand_dims(tf.reshape(tf.range(NMS_TILE_SIZE), [1, -1]) > tf
        .reshape(tf.range(NMS_TILE_SIZE), [-1, 1]), 0)
    iou *= tf.cast(tf.logical_and(mask, iou >= iou_threshold), iou.dtype)
    suppressed_iou, _, _ = tf.while_loop(cond=lambda _iou, loop_condition,
        _iou_sum: loop_condition, body=_self_suppression, loop_vars=[iou,
        tf.constant(True), tf.reduce_sum(input_tensor=iou, axis=[1, 2])])
    suppressed_box = tf.reduce_sum(input_tensor=suppressed_iou, axis=1) > 0
    box_slice *= tf.expand_dims(1.0 - tf.cast(suppressed_box, box_slice.
        dtype), 2)
    mask = tf.reshape(tf.cast(tf.equal(tf.range(num_tiles), idx), boxes.
        dtype), [1, -1, 1, 1])
    boxes = tf.tile(tf.expand_dims(box_slice, [1]), [1, num_tiles, 1, 1]
        ) * mask + tf.reshape(boxes, [batch_size, num_tiles, NMS_TILE_SIZE, 4]
        ) * (1 - mask)
    boxes = tf.reshape(boxes, [batch_size, -1, 4])
    output_size += tf.reduce_sum(input_tensor=tf.cast(tf.reduce_any(
        input_tensor=box_slice > 0, axis=[2]), tf.int32), axis=[1])
    return boxes, iou_threshold, output_size, idx + 1
