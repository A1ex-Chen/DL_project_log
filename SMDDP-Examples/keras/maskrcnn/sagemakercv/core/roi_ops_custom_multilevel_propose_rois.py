def custom_multilevel_propose_rois(scores_outputs, box_outputs, all_anchors,
    image_info, rpn_pre_nms_topn, rpn_post_nms_topn, rpn_nms_threshold,
    rpn_min_size):
    """Proposes RoIs for the second stage nets.

    This proposal op performs the following operations.
    1. propose rois at each level.
    2. collect all proposals.
    3. keep rpn_post_nms_topn proposals by their sorted scores from the highest
       to the lowest.

    Reference:
    https://github.com/facebookresearch/Detectron/blob/master/detectron/ops/collect_and_distribute_fpn_rpn_proposals.py

    Args:
    scores_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4]
    all_anchors: an Anchors object that contains the all anchors.
    image_info: a tensor of shape [batch_size, 5] where the three columns
      encode the input image's [height, width, scale,
      original_height, original_width]. Height and width are for
      the input to the network, not the original image; scale is the scale
      factor used to scale the network input size to the original image size.
      See dataloader.DetectionInputProcessor for details. The last two are
      original height and width. See dataloader.DetectionInputProcessor for
      details.
    rpn_pre_nms_topn: a integer number of top scoring RPN proposals to keep
      before applying NMS. This is *per FPN level* (not total).
    rpn_post_nms_topn: a integer number of top scoring RPN proposals to keep
      after applying NMS. This is the total number of RPN proposals produced.
    rpn_nms_threshold: a float number between 0 and 1 as the NMS threshold
      used on RPN proposals.
    rpn_min_size: a integer number as the minimum proposal height and width as
      both need to be greater than this number. Note that this number is at
      origingal image scale; not scale used during training or inference).
    Returns:
    scores: a tensor with a shape of [batch_size, rpn_post_nms_topn, 1]
      representing the scores of the proposals.
    rois: a tensor with a shape of [batch_size, rpn_post_nms_topn, 4]
      representing the boxes of the proposals. The boxes are in normalized
      coordinates with a form of [ymin, xmin, ymax, xmax].
    """
    with tf.name_scope('proposal'):
        levels = scores_outputs.keys()
        scores = []
        rois = []
        anchor_boxes = all_anchors.get_unpacked_boxes()
        for level in levels:
            logging.debug(
                '[ROI OPs] Using GenerateBoxProposals op... Scope: proposal_%s'
                 % level)
            boxes_per_level, scores_per_level = (tf.image.
                generate_bounding_box_proposals(scores=tf.reshape(tf.
                sigmoid(scores_outputs[level]), scores_outputs[level].shape
                ), bbox_deltas=box_outputs[level], image_info=image_info,
                anchors=anchor_boxes[level], pre_nms_topn=rpn_pre_nms_topn,
                post_nms_topn=rpn_post_nms_topn, nms_threshold=
                rpn_nms_threshold, min_size=rpn_min_size, name=
                'proposal_%s' % level))
            scores.append(scores_per_level)
            rois.append(boxes_per_level)
        scores = tf.concat(scores, axis=1)
        rois = tf.concat(rois, axis=1)
        with tf.name_scope('post_nms_topk'):
            post_nms_num_anchors = scores.shape[1]
            post_nms_topk_limit = (post_nms_num_anchors if 
                post_nms_num_anchors < rpn_post_nms_topn else rpn_post_nms_topn
                )
            top_k_scores, top_k_rois = box_utils.top_k(scores, k=
                post_nms_topk_limit, boxes_list=[rois])
            top_k_rois = top_k_rois[0]
        top_k_scores = tf.stop_gradient(top_k_scores)
        top_k_rois = tf.stop_gradient(top_k_rois)
        return top_k_scores, top_k_rois
