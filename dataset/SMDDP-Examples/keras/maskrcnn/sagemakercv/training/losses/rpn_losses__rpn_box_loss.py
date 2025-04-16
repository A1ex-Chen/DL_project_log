def _rpn_box_loss(self, box_outputs, box_targets, normalizer=1.0, delta=1.0 / 9
    ):
    """Computes box regression loss."""
    with tf.name_scope('rpn_box_loss'):
        mask = tf.not_equal(box_targets, 0.0)
        mask = tf.cast(mask, tf.float32)
        assert mask.dtype == tf.float32
        if self.box_loss_type == 'huber':
            box_loss = _huber_loss(y_true=box_targets, y_pred=box_outputs,
                weights=mask, delta=delta)
            """elif self.box_loss_type == 'giou':
                    box_loss = _giou_loss(y_true=box_targets, y_pred=box_outputs, weights=mask)
                elif self.box_loss_type == 'ciou':
                    box_loss = _ciou_loss(y_true=box_targets, y_pred=box_outputs, weights=mask)"""
        elif self.box_loss_type == 'l1_loss':
            box_loss = _l1_loss(y_true=box_targets, y_pred=box_outputs,
                weights=mask, delta=delta)
        else:
            raise NotImplementedError
        assert box_loss.dtype == tf.float32
        if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
            box_loss /= normalizer
        assert box_loss.dtype == tf.float32
    return box_loss
