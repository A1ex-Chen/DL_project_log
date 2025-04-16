def top_k_boxes(boxes, scores, k):
    """Sort and select top k boxes according to the scores.

  Args:
    boxes: a tensor of shape [batch_size, N, 4] representing the coordiante of
      the boxes. N is the number of boxes per image.
    scores: a tensor of shsape [batch_size, N] representing the socre of the
      boxes.
    k: an integer or a tensor indicating the top k number.

  Returns:
    selected_boxes: a tensor of shape [batch_size, k, 4] representing the
      selected top k box coordinates.
    selected_scores: a tensor of shape [batch_size, k] representing the selected
      top k box scores.
  """
    with tf.name_scope('top_k_boxes'):
        selected_scores, top_k_indices = tf.nn.top_k(scores, k=k, sorted=True)
        batch_size, _ = scores.get_shape().as_list()
        if batch_size == 1:
            selected_boxes = tf.squeeze(tf.gather(boxes, top_k_indices,
                axis=1), axis=1)
        else:
            top_k_indices_shape = tf.shape(top_k_indices)
            batch_indices = tf.expand_dims(tf.range(top_k_indices_shape[0]),
                axis=-1) * tf.ones([1, top_k_indices_shape[-1]], dtype=tf.int32
                )
            gather_nd_indices = tf.stack([batch_indices, top_k_indices],
                axis=-1)
            selected_boxes = tf.gather_nd(boxes, gather_nd_indices)
        return selected_boxes, selected_scores
