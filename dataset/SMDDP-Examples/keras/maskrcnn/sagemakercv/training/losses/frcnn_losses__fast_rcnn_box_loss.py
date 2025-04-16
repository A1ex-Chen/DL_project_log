def _fast_rcnn_box_loss(self, box_outputs, box_targets, class_targets,
    loss_type='huber', normalizer=1.0, delta=1.0):
    """Computes box regression loss."""
    with tf.name_scope('fast_rcnn_box_loss'):
        mask = tf.tile(tf.expand_dims(tf.greater(class_targets, 0), axis=2),
            [1, 1, 4])
        if loss_type == 'huber':
            box_loss = _huber_loss(y_true=box_targets, y_pred=box_outputs,
                weights=mask, delta=delta)
        elif loss_type == 'giou':
            box_loss = _giou_loss(y_true=box_targets, y_pred=box_outputs,
                weights=mask)
        elif loss_type == 'ciou':
            box_loss = _ciou_loss(y_true=box_targets, y_pred=box_outputs,
                weights=mask)
        elif loss_type == 'l1_loss':
            box_loss = _l1_loss(y_true=box_targets, y_pred=box_outputs,
                weights=mask, delta=delta)
        else:
            raise NotImplementedError
        if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
            box_loss /= normalizer
    return box_loss
