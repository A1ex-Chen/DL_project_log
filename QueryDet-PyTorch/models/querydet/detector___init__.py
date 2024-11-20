def __init__(self, cfg):
    super().__init__()
    self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
    self.in_features = cfg.MODEL.RETINANET.IN_FEATURES
    self.query_layer_train = cfg.MODEL.QUERY.Q_FEATURE_TRAIN
    self.layers_whole_test = cfg.MODEL.QUERY.FEATURES_WHOLE_TEST
    self.layers_value_test = cfg.MODEL.QUERY.FEATURES_VALUE_TEST
    self.query_layer_test = cfg.MODEL.QUERY.Q_FEATURE_TEST
    self.focal_loss_alpha = cfg.MODEL.CUSTOM.FOCAL_LOSS_ALPHAS
    self.focal_loss_gamma = cfg.MODEL.CUSTOM.FOCAL_LOSS_GAMMAS
    self.smooth_l1_loss_beta = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA
    self.use_giou_loss = cfg.MODEL.CUSTOM.GIOU_LOSS
    self.cls_weights = cfg.MODEL.CUSTOM.CLS_WEIGHTS
    self.reg_weights = cfg.MODEL.CUSTOM.REG_WEIGHTS
    self.small_obj_scale = cfg.MODEL.QUERY.ENCODE_SMALL_OBJ_SCALE
    self.query_loss_weights = cfg.MODEL.QUERY.QUERY_LOSS_WEIGHT
    self.query_loss_gammas = cfg.MODEL.QUERY.QUERY_LOSS_GAMMA
    self.small_center_dis_coeff = cfg.MODEL.QUERY.ENCODE_CENTER_DIS_COEFF
    self.score_threshold = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
    self.topk_candidates = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
    self.use_soft_nms = cfg.MODEL.CUSTOM.USE_SOFT_NMS
    self.nms_threshold = cfg.MODEL.RETINANET.NMS_THRESH_TEST
    self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
    self.query_infer = cfg.MODEL.QUERY.QUERY_INFER
    self.query_threshold = cfg.MODEL.QUERY.THRESHOLD
    self.query_context = cfg.MODEL.QUERY.CONTEXT
    self.clear_cuda_cache = cfg.MODEL.CUSTOM.CLEAR_CUDA_CACHE
    self.anchor_num = len(cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS[0]) * len(
        cfg.MODEL.ANCHOR_GENERATOR.SIZES[0])
    self.with_cp = cfg.MODEL.CUSTOM.GRADIENT_CHECKPOINT
    assert 'p2' in self.in_features
    self.backbone = build_backbone(cfg)
    if cfg.MODEL.CUSTOM.HEAD_BN:
        self.det_head = dh.RetinaNetHead_3x3_MergeBN(cfg, 256, 256, 4, self
            .anchor_num)
        self.query_head = dh.Head_3x3_MergeBN(256, 256, 4, 1)
    else:
        self.det_head = dh.RetinaNetHead_3x3(cfg, 256, 256, 4, self.anchor_num)
        self.query_head = dh.Head_3x3(256, 256, 4, 1)
    self.qInfer = qf.QueryInfer(9, self.num_classes, self.query_threshold,
        self.query_context)
    backbone_shape = self.backbone.output_shape()
    all_det_feature_shapes = [backbone_shape[f] for f in self.in_features]
    self.anchor_generator = build_anchor_generator(cfg, all_det_feature_shapes)
    self.query_anchor_generator = AnchorGeneratorWithCenter(sizes=[128],
        aspect_ratios=[1.0], strides=[(2 ** (x + 2)) for x in self.
        query_layer_train], offset=0.5)
    self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.
        BBOX_REG_WEIGHTS)
    self.soft_nmser = SoftNMSer(cfg.MODEL.CUSTOM.SOFT_NMS_METHOD, cfg.MODEL
        .CUSTOM.SOFT_NMS_SIGMA, cfg.MODEL.CUSTOM.SOFT_NMS_THRESHOLD, cfg.
        MODEL.CUSTOM.SOFT_NMS_PRUND)
    if cfg.MODEL.CUSTOM.USE_LOOP_MATCHER:
        self.matcher = LoopMatcher(cfg.MODEL.RETINANET.IOU_THRESHOLDS, cfg.
            MODEL.RETINANET.IOU_LABELS, allow_low_quality_matches=True)
    else:
        self.matcher = Matcher(cfg.MODEL.RETINANET.IOU_THRESHOLDS, cfg.
            MODEL.RETINANET.IOU_LABELS, allow_low_quality_matches=True)
    self.register_buffer('pixel_mean', torch.Tensor(cfg.MODEL.PIXEL_MEAN).
        view(-1, 1, 1))
    self.register_buffer('pixel_std', torch.Tensor(cfg.MODEL.PIXEL_STD).
        view(-1, 1, 1))
    self.loss_normalizer = 100
    self.loss_normalizer_momentum = 0.9
