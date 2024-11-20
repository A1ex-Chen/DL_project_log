def __init__(self, output_shape, num_class=2, num_input_features=4,
    voxelization_name='BV', vfe_num_filters=[32, 128], with_distance=False,
    tdbn_name='tDBN_1', tdbn_filters_d1=[64], tdbn_filters_d2=[64, 64],
    det_net_name='det_net', det_net_layer_nums=[3, 5, 5],
    det_net_layer_strides=[2, 2, 2], det_net_num_filters=[128, 128, 256],
    det_net_upsample_strides=[1, 2, 4], det_net_num_upsample_filters=[256, 
    256, 256], use_norm=True, use_groupnorm=False, num_groups=32,
    use_direction_classifier=True, use_sigmoid_score=False,
    encode_background_as_zeros=True, use_rotate_nms=True, multiclass_nms=
    False, nms_score_threshold=0.5, nms_pre_max_size=1000,
    nms_post_max_size=20, nms_iou_threshold=0.1, target_assigner=None,
    lidar_only=False, cls_loss_weight=1.0, loc_loss_weight=1.0,
    pos_cls_weight=1.0, neg_cls_weight=1.0, direction_loss_weight=1.0,
    loss_norm_type=LossNormType.NormByNumPositives, encode_rad_error_by_sin
    =False, loc_loss_ftor=None, cls_loss_ftor=None, name='model_net'):
    super().__init__()
    self.name = name
    self._num_class = num_class
    self._use_rotate_nms = use_rotate_nms
    self._multiclass_nms = multiclass_nms
    self._nms_score_threshold = nms_score_threshold
    self._nms_pre_max_size = nms_pre_max_size
    self._nms_post_max_size = nms_post_max_size
    self._nms_iou_threshold = nms_iou_threshold
    self._use_sigmoid_score = use_sigmoid_score
    self._encode_background_as_zeros = encode_background_as_zeros
    self._use_direction_classifier = use_direction_classifier
    self._total_forward_time = 0.0
    self._total_postprocess_time = 0.0
    self._total_inference_count = 0
    self._num_input_features = num_input_features
    self._box_coder = target_assigner.box_coder
    self._lidar_only = lidar_only
    self.target_assigner = target_assigner
    self._pos_cls_weight = pos_cls_weight
    self._neg_cls_weight = neg_cls_weight
    self._encode_rad_error_by_sin = encode_rad_error_by_sin
    self._loss_norm_type = loss_norm_type
    self._dir_loss_ftor = WeightedSoftmaxClassificationLoss()
    self._loc_loss_ftor = loc_loss_ftor
    self._cls_loss_ftor = cls_loss_ftor
    self._direction_loss_weight = direction_loss_weight
    self._cls_loss_weight = cls_loss_weight
    self._loc_loss_weight = loc_loss_weight
    voxelization_class_dict = {'VEF': voxelization.VoxelFeatureExtractor,
        'BV': voxelization.BinaryVoxel}
    voxelization_class = voxelization_class_dict[voxelization_name]
    self.voxel_feature_extractor = voxelization_class(num_input_features,
        use_norm, num_filters=vfe_num_filters, with_distance=with_distance)
    tdbn_class_dict = {'tDBN_1': tDBN.tDBN_1, 'tDBN_2': tDBN.tDBN_2,
        'tDBN_bv_1': tDBN.tDBN_bv_1, 'tDBN_bv_2': tDBN.tDBN_bv_2}
    tdbn_class = tdbn_class_dict[tdbn_name]
    self.tdbn_feature_extractor = tdbn_class(output_shape, use_norm,
        num_filters_down1=tdbn_filters_d1, num_filters_down2=tdbn_filters_d2)
    det_net_class_dict = {'det_net': det_net.det_net, 'det_net_2': det_net.
        det_net_2}
    det_net_class = det_net_class_dict[det_net_name]
    self.det_net = det_net_class(use_norm=True, num_class=num_class,
        layer_nums=det_net_layer_nums, layer_strides=det_net_layer_strides,
        num_filters=det_net_num_filters, upsample_strides=
        det_net_upsample_strides, num_upsample_filters=
        det_net_num_upsample_filters, num_anchor_per_loc=target_assigner.
        num_anchors_per_location, encode_background_as_zeros=
        encode_background_as_zeros, use_direction_classifier=
        use_direction_classifier, use_groupnorm=use_groupnorm, num_groups=
        num_groups, box_code_size=target_assigner.box_coder.code_size)
    self.det_net_acc = metrics.Accuracy(dim=-1, encode_background_as_zeros=
        encode_background_as_zeros)
    self.det_net_precision = metrics.Precision(dim=-1)
    self.det_net_recall = metrics.Recall(dim=-1)
    self.det_net_metrics = metrics.PrecisionRecall(dim=-1, thresholds=[0.1,
        0.3, 0.5, 0.7, 0.8, 0.9, 0.95], use_sigmoid_score=use_sigmoid_score,
        encode_background_as_zeros=encode_background_as_zeros)
    self.det_net_cls_loss = metrics.Scalar()
    self.det_net_loc_loss = metrics.Scalar()
    self.det_net_total_loss = metrics.Scalar()
    self.register_buffer('global_step', torch.LongTensor(1).zero_())
