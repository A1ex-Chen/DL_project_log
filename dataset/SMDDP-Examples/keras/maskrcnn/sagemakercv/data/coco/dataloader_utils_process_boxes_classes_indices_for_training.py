def process_boxes_classes_indices_for_training(data,
    skip_crowd_during_training, use_category, use_instance_mask):
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']
    classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
    indices = None
    instance_masks = None
    if not use_category:
        classes = tf.cast(tf.greater(classes, 0), dtype=tf.float32)
    if skip_crowd_during_training:
        indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
        classes = tf.gather_nd(classes, indices)
        boxes = tf.gather_nd(boxes, indices)
        if use_instance_mask:
            instance_masks = tf.gather_nd(data['groundtruth_instance_masks'
                ], indices)
    return boxes, classes, indices, instance_masks
