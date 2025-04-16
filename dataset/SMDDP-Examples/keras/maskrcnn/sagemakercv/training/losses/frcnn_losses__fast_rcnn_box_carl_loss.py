def _fast_rcnn_box_carl_loss(self, box_outputs, box_targets, class_targets,
    class_outputs, beta=0.2, gamma=1.0, num_classes=91, loss_type='huber',
    normalizer=1.0, delta=1.0):
    """Computes classification aware box regression loss."""
    with tf.name_scope('fast_rcnn_carl_loss'):
        mask = tf.tile(tf.expand_dims(tf.greater(class_targets, 0), axis=2),
            [1, 1, 4])
        if loss_type == 'huber':
            box_loss = _huber_loss(y_true=box_targets, y_pred=box_outputs,
                weights=mask, delta=delta, reduction=ReductionV2.NONE)
        elif loss_type == 'giou':
            box_loss = _giou_loss(y_true=box_targets, y_pred=box_outputs,
                weights=mask, reduction='none')
            box_loss = tf.reshape(box_loss, [-1, 512, 4])
        elif loss_type == 'ciou':
            box_loss = _ciou_loss(y_true=box_targets, y_pred=box_outputs,
                weights=mask)
        else:
            raise NotImplementedError
        oh_targets = tf.one_hot(class_targets, depth=num_classes)
        class_scores = tf.nn.softmax(class_outputs, axis=-1)
        top_class_score = class_scores * oh_targets
        pos_cls_score = top_class_score[:, :, 1:]
        pos_cls_score = tf.reduce_max(pos_cls_score, axis=-1)
        carl_loss_weights = beta + (1.0 - beta) * pos_cls_score
        carl_loss_weights = tf.where(pos_cls_score > 0.0, carl_loss_weights,
            tf.zeros_like(carl_loss_weights))
        num_pos = tf.math.count_nonzero(class_targets, dtype=
            carl_loss_weights.dtype)
        weight_ratio = tf.math.divide_no_nan(num_pos, tf.reduce_sum(
            carl_loss_weights))
        carl_loss_weights *= weight_ratio
        loss_carl = tf.reduce_sum(box_loss * tf.expand_dims(
            carl_loss_weights, -1))
        loss_bbox = tf.reduce_sum(box_loss)
        assert loss_carl.dtype == tf.float32
        regression_loss = loss_carl + loss_bbox
        if isinstance(normalizer, tf.Tensor) or normalizer != 1.0:
            regression_loss /= normalizer
        return regression_loss
