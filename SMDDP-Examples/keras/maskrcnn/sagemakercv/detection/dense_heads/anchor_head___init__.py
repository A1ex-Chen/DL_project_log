def __init__(self, anchor_generator_cfg=dict(min_level=2, max_level=6,
    num_scales=1, aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
    anchor_scale=8.0, image_size=(832, 1344)), roi_proposal_cfg=dict(
    train_cfg=dict(rpn_pre_nms_topn=2000, rpn_post_nms_topn=1000,
    rpn_nms_threshold=0.7), test_cfg=dict(rpn_pre_nms_topn=1000,
    rpn_post_nms_topn=1000, rpn_nms_threshold=0.7), rpn_min_size=0.0,
    use_custom_box_proposals_op=True, use_batched_nms=False,
    bbox_reg_weights=None), num_classes=1, feat_channels=256, trainable=True):
    super(AnchorHead, self).__init__()
    self.num_classes = num_classes
    self.feat_channels = feat_channels
    self.cls_out_channels = num_classes
    self.anchor_generator = AnchorGenerator(**anchor_generator_cfg)
    self.roi_proposal = ProposeROIs(**roi_proposal_cfg)
    self.trainable = trainable
    self._init_layers()
