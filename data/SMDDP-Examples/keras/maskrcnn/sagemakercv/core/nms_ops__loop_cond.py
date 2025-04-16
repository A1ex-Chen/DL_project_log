def _loop_cond(unused_boxes, unused_threshold, output_size, idx):
    return tf.logical_and(tf.reduce_min(output_size) < max_output_size, idx <
        num_boxes // NMS_TILE_SIZE)
