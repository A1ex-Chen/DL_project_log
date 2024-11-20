def proposal_label_op(boxes, gt_boxes, gt_labels, batch_size_per_im=512,
    fg_fraction=0.25, fg_thresh=0.5, bg_thresh_hi=0.5, bg_thresh_lo=0.0):
    """Assigns the proposals with ground truth labels and performs subsmpling.

    Given proposal `boxes`, `gt_boxes`, and `gt_labels`, the function uses the
    following algorithm to generate the final `batch_size_per_im` RoIs.
    1. Calculates the IoU between each proposal box and each gt_boxes.
    2. Assigns each proposal box with a ground truth class and box label by
     choosing the largest overlap.
    3. Samples `batch_size_per_im` boxes from all proposal boxes, and returns
     box_targets, class_targets, and RoIs.
    The reference implementations of #1 and #2 are here:
    https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/json_dataset.py
    The reference implementation of #3 is here:
    https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py

    Args:
    boxes: a tensor with a shape of [batch_size, N, 4]. N is the number of
      proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
      last dimension is the pixel coordinates of scaled images in
      [ymin, xmin, ymax, xmax] form.
    gt_boxes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. This
      tensor might have paddings with a value of -1. The coordinates of gt_boxes
      are in the pixel coordinates of the scaled image.
    gt_labels: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES]. This
      tensor might have paddings with a value of -1.
    batch_size_per_im: a integer represents RoI minibatch size per image.
    fg_fraction: a float represents the target fraction of RoI minibatch that
      is labeled foreground (i.e., class > 0).
    fg_thresh: a float represents the overlap threshold for an RoI to be
      considered foreground (if >= fg_thresh).
    bg_thresh_hi: a float represents the overlap threshold for an RoI to be
      considered background (class = 0 if overlap in [LO, HI)).
    bg_thresh_lo: a float represents the overlap threshold for an RoI to be
      considered background (class = 0 if overlap in [LO, HI)).
    Returns:
    box_targets: a tensor with a shape of [batch_size, K, 4]. The tensor
      contains the ground truth pixel coordinates of the scaled images for each
      roi. K is the number of sample RoIs (e.g., batch_size_per_im).
    class_targets: a integer tensor with a shape of [batch_size, K]. The tensor
      contains the ground truth class for each roi.
    rois: a tensor with a shape of [batch_size, K, 4], representing the
      coordinates of the selected RoI.
    proposal_to_label_map: a tensor with a shape of [batch_size, K]. This tensor
      keeps the mapping between proposal to labels. proposal_to_label_map[i]
      means the index of the ground truth instance for the i-th proposal.
    """
    with tf.name_scope('proposal_label'):
        batch_size = boxes.shape[0]
        boxes = tf.concat([boxes, gt_boxes], axis=1)
        iou = box_utils.bbox_overlap(boxes, gt_boxes)
        (pre_sample_box_targets, pre_sample_class_targets, max_overlap,
            proposal_to_label_map) = _add_class_assignments(iou, gt_boxes,
            gt_labels)
        positives = tf.greater(max_overlap, fg_thresh * tf.ones_like(
            max_overlap))
        negatives = tf.logical_and(tf.greater_equal(max_overlap, 
            bg_thresh_lo * tf.ones_like(max_overlap)), tf.less(max_overlap,
            bg_thresh_hi * tf.ones_like(max_overlap)))
        pre_sample_class_targets = tf.where(negatives, tf.zeros_like(
            pre_sample_class_targets), pre_sample_class_targets)
        proposal_to_label_map = tf.where(negatives, tf.zeros_like(
            proposal_to_label_map), proposal_to_label_map)
        ignore_mask = tf.less(tf.reduce_min(input_tensor=iou, axis=2), tf.
            zeros_like(max_overlap))
        labels = positives
        pos_or_neg = tf.logical_or(positives, negatives)
        indicator = tf.logical_and(pos_or_neg, tf.logical_not(ignore_mask))
        all_samples = []
        sampler = (balanced_positive_negative_sampler.
            BalancedPositiveNegativeSampler(positive_fraction=fg_fraction,
            is_static=True))
        for i in range(batch_size):
            samples = sampler.subsample(indicator[i], batch_size_per_im,
                labels[i])
            all_samples.append(samples)
        all_samples = tf.stack([all_samples], axis=0)[0]
        _, samples_indices = tf.nn.top_k(tf.cast(all_samples, dtype=tf.
            int32), k=batch_size_per_im, sorted=True)
        samples_indices = tf.reshape(samples_indices + tf.expand_dims(tf.
            range(batch_size) * tf.shape(input=boxes)[1], 1), [-1])
        rois = tf.reshape(tf.gather(tf.reshape(boxes, [-1, 4]),
            samples_indices), [batch_size, -1, 4])
        class_targets = tf.reshape(tf.gather(tf.reshape(
            pre_sample_class_targets, [-1, 1]), samples_indices), [
            batch_size, -1])
        sample_box_targets = tf.reshape(tf.gather(tf.reshape(
            pre_sample_box_targets, [-1, 4]), samples_indices), [batch_size,
            -1, 4])
        sample_proposal_to_label_map = tf.reshape(tf.gather(tf.reshape(
            proposal_to_label_map, [-1, 1]), samples_indices), [batch_size, -1]
            )
    return (sample_box_targets, class_targets, rois,
        sample_proposal_to_label_map)
