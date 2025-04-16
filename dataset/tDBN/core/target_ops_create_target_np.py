def create_target_np(all_anchors, gt_boxes, similarity_fn, box_encoding_fn,
    prune_anchor_fn=None, gt_classes=None, matched_threshold=0.6,
    unmatched_threshold=0.45, bbox_inside_weight=None, positive_fraction=
    None, rpn_batch_size=300, norm_by_num_examples=False, box_code_size=7):
    """Modified from FAIR detectron.
    Args:
        all_anchors: [num_of_anchors, box_ndim] float tensor.
        gt_boxes: [num_gt_boxes, box_ndim] float tensor.
        similarity_fn: a function, accept anchors and gt_boxes, return
            similarity matrix(such as IoU).
        box_encoding_fn: a function, accept gt_boxes and anchors, return
            box encodings(offsets).
        prune_anchor_fn: a function, accept anchors, return indices that
            indicate valid anchors.
        gt_classes: [num_gt_boxes] int tensor. indicate gt classes, must
            start with 1.
        matched_threshold: float, iou greater than matched_threshold will
            be treated as positives.
        unmatched_threshold: float, iou smaller than unmatched_threshold will
            be treated as negatives.
        bbox_inside_weight: unused
        positive_fraction: [0-1] float or None. if not None, we will try to
            keep ratio of pos/neg equal to positive_fraction when sample.
            if there is not enough positives, it fills the rest with negatives
        rpn_batch_size: int. sample size
        norm_by_num_examples: bool. norm box_weight by number of examples, but
            I recommend to do this outside.
    Returns:
        labels, bbox_targets, bbox_outside_weights
    """
    total_anchors = all_anchors.shape[0]
    if prune_anchor_fn is not None:
        inds_inside = prune_anchor_fn(all_anchors)
        anchors = all_anchors[inds_inside, :]
        if not isinstance(matched_threshold, float):
            matched_threshold = matched_threshold[inds_inside]
        if not isinstance(unmatched_threshold, float):
            unmatched_threshold = unmatched_threshold[inds_inside]
    else:
        anchors = all_anchors
        inds_inside = None
    num_inside = len(inds_inside) if inds_inside is not None else total_anchors
    box_ndim = all_anchors.shape[1]
    logger.debug('total_anchors: {}'.format(total_anchors))
    logger.debug('inds_inside: {}'.format(num_inside))
    logger.debug('anchors.shape: {}'.format(anchors.shape))
    if gt_classes is None:
        gt_classes = np.ones([gt_boxes.shape[0]], dtype=np.int32)
    labels = np.empty((num_inside,), dtype=np.int32)
    gt_ids = np.empty((num_inside,), dtype=np.int32)
    labels.fill(-1)
    gt_ids.fill(-1)
    if len(gt_boxes) > 0 and anchors.shape[0] > 0:
        anchor_by_gt_overlap = similarity_fn(anchors, gt_boxes)
        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
        anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_inside),
            anchor_to_gt_argmax]
        gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
        gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, np.
            arange(anchor_by_gt_overlap.shape[1])]
        empty_gt_mask = gt_to_anchor_max == 0
        gt_to_anchor_max[empty_gt_mask] = -1
        anchors_with_max_overlap = np.where(anchor_by_gt_overlap ==
            gt_to_anchor_max)[0]
        gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
        labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
        gt_ids[anchors_with_max_overlap] = gt_inds_force
        pos_inds = anchor_to_gt_max >= matched_threshold
        gt_inds = anchor_to_gt_argmax[pos_inds]
        labels[pos_inds] = gt_classes[gt_inds]
        gt_ids[pos_inds] = gt_inds
        bg_inds = np.where(anchor_to_gt_max < unmatched_threshold)[0]
    else:
        bg_inds = np.arange(num_inside)
    fg_inds = np.where(labels > 0)[0]
    fg_max_overlap = None
    if len(gt_boxes) > 0 and anchors.shape[0] > 0:
        fg_max_overlap = anchor_to_gt_max[fg_inds]
    gt_pos_ids = gt_ids[fg_inds]
    if positive_fraction is not None:
        num_fg = int(positive_fraction * rpn_batch_size)
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(fg_inds, size=len(fg_inds) - num_fg,
                replace=False)
            labels[disable_inds] = -1
            fg_inds = np.where(labels > 0)[0]
        num_bg = rpn_batch_size - np.sum(labels > 0)
        if len(bg_inds) > num_bg:
            enable_inds = bg_inds[npr.randint(len(bg_inds), size=num_bg)]
            labels[enable_inds] = 0
        bg_inds = np.where(labels == 0)[0]
    elif len(gt_boxes) == 0 or anchors.shape[0] == 0:
        labels[:] = 0
    else:
        labels[bg_inds] = 0
        labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
    bbox_targets = np.zeros((num_inside, box_code_size), dtype=all_anchors.
        dtype)
    if len(gt_boxes) > 0 and anchors.shape[0] > 0:
        bbox_targets[fg_inds, :] = box_encoding_fn(gt_boxes[
            anchor_to_gt_argmax[fg_inds], :], anchors[fg_inds, :])
    bbox_outside_weights = np.zeros((num_inside,), dtype=all_anchors.dtype)
    if norm_by_num_examples:
        num_examples = np.sum(labels >= 0)
        num_examples = np.maximum(1.0, num_examples)
        bbox_outside_weights[labels > 0] = 1.0 / num_examples
    else:
        bbox_outside_weights[labels > 0] = 1.0
    if inds_inside is not None:
        labels = unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = unmap(bbox_outside_weights, total_anchors,
            inds_inside, fill=0)
    ret = {'labels': labels, 'bbox_targets': bbox_targets,
        'bbox_outside_weights': bbox_outside_weights,
        'assigned_anchors_overlap': fg_max_overlap, 'positive_gt_id':
        gt_pos_ids}
    if inds_inside is not None:
        ret['assigned_anchors_inds'] = inds_inside[fg_inds]
    else:
        ret['assigned_anchors_inds'] = fg_inds
    return ret
