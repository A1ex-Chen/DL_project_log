def __call__(self, rpn_score_outputs, rpn_box_outputs, image_info, anchors,
    training=True):
    cfg = self.train_cfg if training else self.test_cfg
    args = {'scores_outputs': rpn_score_outputs, 'box_outputs':
        rpn_box_outputs, 'all_anchors': anchors, 'image_info': image_info,
        'rpn_min_size': self.rpn_min_size}
    args.update(cfg)
    if not self.use_custom_box_proposals_op:
        args.update({'use_batched_nms': self.use_batched_nms,
            'bbox_reg_weights': self.bbox_reg_weights})
    roi_op = (custom_multilevel_propose_rois if self.
        use_custom_box_proposals_op else multilevel_propose_rois)
    rpn_box_scores, rpn_box_rois = roi_op(**args)
    rpn_box_rois = tf.cast(rpn_box_rois, dtype=tf.float32)
    if training:
        rpn_box_rois = tf.stop_gradient(rpn_box_rois)
        rpn_box_scores = tf.stop_gradient(rpn_box_scores)
    return rpn_box_rois, rpn_box_scores
