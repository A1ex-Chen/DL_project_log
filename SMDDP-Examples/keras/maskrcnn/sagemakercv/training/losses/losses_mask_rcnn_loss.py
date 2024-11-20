def mask_rcnn_loss(mask_outputs, mask_targets, select_class_targets, params):
    """Computes the mask loss of Mask-RCNN.
    This function implements the mask loss of Mask-RCNN. As the `mask_outputs`
    produces `num_classes` masks for each RoI, the reference model expands
    `mask_targets` to match the shape of `mask_outputs` and selects only the
    target that the RoI has a maximum overlap.
    (Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/mask_rcnn.py)
    Instead, this implementation selects the `mask_outputs` by the `class_targets`
    so that it doesn't expand `mask_targets`. Note that the selection logic is
    done in the post-processing of mask_rcnn_fn in mask_rcnn_architecture.py.
    Args:
    mask_outputs: a float tensor representing the prediction for each mask,
      with a shape of
      [batch_size, num_masks, mask_height, mask_width].
    mask_targets: a float tensor representing the binary mask of ground truth
      labels for each mask with a shape of
      [batch_size, num_masks, mask_height, mask_width].
    select_class_targets: a tensor with a shape of [batch_size, num_masks],
      representing the foreground mask targets.
    params: the dictionary including training parameters specified in
      default_haprams function in this file.
    Returns:
    mask_loss: a float tensor representing total mask loss.
    """
    with tf.name_scope('mask_loss'):
        batch_size, num_masks, mask_height, mask_width = (mask_outputs.
            get_shape().as_list())
        weights = tf.tile(tf.reshape(tf.greater(select_class_targets, 0), [
            batch_size, num_masks, 1, 1]), [1, 1, mask_height, mask_width])
        weights = tf.cast(weights, tf.float32)
        loss = _sigmoid_cross_entropy(multi_class_labels=mask_targets,
            logits=mask_outputs, weights=weights, sum_by_non_zeros_weights=
            True, label_smoothing=params['label_smoothing'])
        mrcnn_loss = params['mrcnn_weight_loss_mask'] * loss
        return mrcnn_loss
