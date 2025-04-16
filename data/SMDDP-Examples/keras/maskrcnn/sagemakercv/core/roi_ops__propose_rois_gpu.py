def _propose_rois_gpu(scores, boxes, anchor_boxes, height, width, scale,
    rpn_pre_nms_topn, rpn_post_nms_topn, rpn_nms_threshold, rpn_min_size,
    bbox_reg_weights):
    """Proposes RoIs giva group of candidates (GPU version).

    Args:
    scores: a tensor with a shape of [batch_size, num_boxes].
    boxes: a tensor with a shape of [batch_size, num_boxes, 4],
      in the encoded form.
    anchor_boxes: an Anchors object that contains the anchors with a shape of
      [batch_size, num_boxes, 4].
    height: a tensor of shape [batch_size, 1, 1] representing the image height.
    width: a tensor of shape [batch_size, 1, 1] representing the image width.
    scale: a tensor of shape [batch_size, 1, 1] representing the image scale.
    rpn_pre_nms_topn: a integer number of top scoring RPN proposals to keep
      before applying NMS. This is *per FPN level* (not total).
    rpn_post_nms_topn: a integer number of top scoring RPN proposals to keep
      after applying NMS. This is the total number of RPN proposals produced.
    rpn_nms_threshold: a float number between 0 and 1 as the NMS threshold
      used on RPN proposals.
    rpn_min_size: a integer number as the minimum proposal height and width as
      both need to be greater than this number. Note that this number is at
      origingal image scale; not scale used during training or inference).
    bbox_reg_weights: None or a list of four integer specifying the weights used
      when decoding the box.

    Returns:
    scores: a tensor with a shape of [batch_size, rpn_post_nms_topn, 1]
      representing the scores of the proposals. It has same dtype as input
      scores.
    boxes: a tensor with a shape of [batch_size, rpn_post_nms_topn, 4]
      represneting the boxes of the proposals. The boxes are in normalized
      coordinates with a form of [ymin, xmin, ymax, xmax]. It has same dtype as
      input boxes.
    """
    batch_size, num_boxes = scores.get_shape().as_list()
    topk_limit = min(num_boxes, rpn_pre_nms_topn)
    boxes = box_utils.decode_boxes(boxes, anchor_boxes, bbox_reg_weights)
    boxes = box_utils.clip_boxes(boxes, (height, width))
    if rpn_min_size > 0.0:
        boxes, scores = box_utils.filter_boxes_2(boxes, tf.expand_dims(
            scores, axis=-1), rpn_min_size, height, width, scale)
        scores = tf.squeeze(scores, axis=-1)
    post_nms_topk_limit = (topk_limit if topk_limit < rpn_post_nms_topn else
        rpn_post_nms_topn)
    if rpn_nms_threshold > 0:
        pre_nms_boxes = box_utils.to_normalized_coordinates(boxes, height,
            width)
        pre_nms_boxes = tf.reshape(pre_nms_boxes, [batch_size, num_boxes, 1, 4]
            )
        pre_nms_scores = tf.reshape(scores, [batch_size, num_boxes, 1])
        with tf.device('CPU:0'):
            boxes, scores, _, _ = tf.image.combined_non_max_suppression(
                pre_nms_boxes, pre_nms_scores, max_output_size_per_class=
                topk_limit, max_total_size=post_nms_topk_limit,
                iou_threshold=rpn_nms_threshold, score_threshold=0.0,
                pad_per_class=False)
        boxes = box_utils.to_absolute_coordinates(boxes, height, width)
    else:
        scores, boxes = box_utils.top_k(scores, k=post_nms_topk_limit,
            boxes_list=[boxes])
        boxes = boxes[0]
    return scores, boxes
