def _cross_suppression(boxes, box_slice, iou_threshold, inner_idx):
    batch_size = tf.shape(boxes)[0]
    new_slice = tf.slice(boxes, [0, inner_idx * NMS_TILE_SIZE, 0], [
        batch_size, NMS_TILE_SIZE, 4])
    iou = box_utils.bbox_overlap(new_slice, box_slice)
    ret_slice = tf.expand_dims(tf.cast(tf.reduce_all(iou < iou_threshold, [
        1]), box_slice.dtype), 2) * box_slice
    return boxes, ret_slice, iou_threshold, inner_idx + 1
