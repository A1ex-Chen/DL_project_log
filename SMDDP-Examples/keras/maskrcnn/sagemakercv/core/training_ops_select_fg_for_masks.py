def select_fg_for_masks(class_targets, box_targets, boxes,
    proposal_to_label_map, max_num_fg=128):
    """Selects the fore ground objects for mask branch during training.

    Args:
    class_targets: a tensor of shape [batch_size, num_boxes]  representing the
      class label for each box.
    box_targets: a tensor with a shape of [batch_size, num_boxes, 4]. The tensor
      contains the ground truth pixel coordinates of the scaled images for each
      roi.
    boxes: A 3-D Tensor of shape [batch_size, num_boxes, 4]. Each row
      represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
    proposal_to_label_map: a tensor with a shape of [batch_size, num_boxes].
      This tensor keeps the mapping between proposal to labels.
      proposal_to_label_map[i] means the index of the ground truth instance for
      the i-th proposal.
    max_num_fg: a integer represents the number of masks per image.
    Returns:
    class_targets, boxes, proposal_to_label_map, box_targets that have
    foreground objects.
    """
    batch_size = boxes.shape[0]
    _, fg_indices = tf.nn.top_k(tf.cast(tf.greater(class_targets, 0), dtype
        =tf.float32), k=max_num_fg)
    indices = tf.reshape(fg_indices + tf.expand_dims(tf.range(batch_size) *
        tf.shape(input=class_targets)[1], 1), [-1])
    fg_class_targets = tf.reshape(tf.gather(tf.reshape(class_targets, [-1, 
        1]), indices), [batch_size, -1])
    fg_box_targets = tf.reshape(tf.gather(tf.reshape(box_targets, [-1, 4]),
        indices), [batch_size, -1, 4])
    fg_box_rois = tf.reshape(tf.gather(tf.reshape(boxes, [-1, 4]), indices),
        [batch_size, -1, 4])
    fg_proposal_to_label_map = tf.reshape(tf.gather(tf.reshape(
        proposal_to_label_map, [-1, 1]), indices), [batch_size, -1])
    return (fg_class_targets, fg_box_targets, fg_box_rois,
        fg_proposal_to_label_map)
