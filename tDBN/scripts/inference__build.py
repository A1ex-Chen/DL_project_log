def _build(self):
    config = self.config
    input_cfg = config.eval_input_reader
    model_cfg = config.model.tDBN
    train_cfg = config.train_config
    batch_size = 1
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    grid_size = voxel_generator.grid_size
    self.voxel_generator = voxel_generator
    vfe_num_filters = list(model_cfg.voxel_feature_extractor.num_filters)
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
        bv_range, box_coder)
    self.target_assigner = target_assigner
    out_size_factor = model_cfg.rpn.layer_strides[0
        ] // model_cfg.rpn.upsample_strides[0]
    self.net = tDBN_builder.build(model_cfg, voxel_generator, target_assigner)
    self.net.cuda().eval()
    if train_cfg.enable_mixed_precision:
        self.net.half()
        self.net.metrics_to_float()
        self.net.convert_norm_to_float(self.net)
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    ret = target_assigner.generate_anchors(feature_map_size)
    anchors = ret['anchors']
    anchors = anchors.reshape([-1, 7])
    matched_thresholds = ret['matched_thresholds']
    unmatched_thresholds = ret['unmatched_thresholds']
    anchors_bv = box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
    self.anchor_cache = {'anchors': anchors, 'anchors_bv': anchors_bv,
        'matched_thresholds': matched_thresholds, 'unmatched_thresholds':
        unmatched_thresholds}
