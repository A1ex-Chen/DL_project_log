def __init__(self, train_cfg=dict(rpn_pre_nms_topn=2000, rpn_post_nms_topn=
    1000, rpn_nms_threshold=0.7), test_cfg=dict(rpn_pre_nms_topn=1000,
    rpn_post_nms_topn=1000, rpn_nms_threshold=0.7), rpn_min_size=0.0,
    use_custom_box_proposals_op=True, use_batched_nms=False,
    bbox_reg_weights=None):
    self.train_cfg = train_cfg
    self.test_cfg = test_cfg
    self.rpn_min_size = rpn_min_size
    self.use_custom_box_proposals_op = use_custom_box_proposals_op
    self.use_batched_nms = use_batched_nms
    self.bbox_reg_weights = bbox_reg_weights
