def __init__(self, num_classes: int, backbone: Backbone, anchor_ratios:
    List[Tuple[int, int]], anchor_sizes: List[int], train_rpn_pre_nms_top_n:
    int, train_rpn_post_nms_top_n: int, eval_rpn_pre_nms_top_n: int,
    eval_rpn_post_nms_top_n: int, num_anchor_samples_per_batch: int,
    num_proposal_samples_per_batch: int, num_detections_per_image: int,
    anchor_smooth_l1_loss_beta: float, proposal_smooth_l1_loss_beta: float,
    proposal_nms_threshold: float, detection_nms_threshold: float):
    super().__init__(num_classes, backbone, anchor_ratios, anchor_sizes,
        train_rpn_pre_nms_top_n, train_rpn_post_nms_top_n,
        eval_rpn_pre_nms_top_n, eval_rpn_post_nms_top_n,
        num_anchor_samples_per_batch, num_proposal_samples_per_batch,
        num_detections_per_image, anchor_smooth_l1_loss_beta,
        proposal_smooth_l1_loss_beta, proposal_nms_threshold,
        detection_nms_threshold)
    if isinstance(backbone, MobileNet_v3_Small):
        backbone_name = Backbone.Name.MOBILENET_V3_SMALL.value
        backbone = mobilenet_backbone(backbone_name, pretrained=backbone.
            pretrained, fpn=True, trainable_layers=5 - backbone.
            num_frozen_levels)
        anchor_sizes = (tuple(it for it in anchor_sizes),) * len(anchor_ratios)
    elif isinstance(backbone, MobileNet_v3_Large):
        backbone_name = Backbone.Name.MOBILENET_V3_LARGE.value
        backbone = mobilenet_backbone(backbone_name, pretrained=backbone.
            pretrained, fpn=True, trainable_layers=5 - backbone.
            num_frozen_levels)
        anchor_sizes = (tuple(it for it in anchor_sizes),) * len(anchor_ratios)
    elif isinstance(backbone, ResNet50):
        backbone_name = Backbone.Name.RESNET50.value
        backbone = resnet_fpn_backbone(backbone_name, pretrained=backbone.
            pretrained, trainable_layers=5 - backbone.num_frozen_levels)
        anchor_sizes = tuple((it,) for it in anchor_sizes)
    else:
        raise ValueError(f'Unsupported backbone for this algorithm')
    min_size, max_size, image_mean, image_std = None, None, None, None
    aspect_ratios = (tuple(it[0] / it[1] for it in anchor_ratios),) * len(
        anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    self.net = torchvision.models.detection.FasterRCNN(backbone,
        num_classes, min_size, max_size, image_mean, image_std,
        rpn_anchor_generator, rpn_pre_nms_top_n_train=
        train_rpn_pre_nms_top_n, rpn_pre_nms_top_n_test=
        eval_rpn_pre_nms_top_n, rpn_post_nms_top_n_train=
        train_rpn_post_nms_top_n, rpn_post_nms_top_n_test=
        eval_rpn_post_nms_top_n, rpn_nms_thresh=proposal_nms_threshold,
        rpn_batch_size_per_image=num_anchor_samples_per_batch,
        box_nms_thresh=detection_nms_threshold, box_detections_per_img=
        num_detections_per_image, box_batch_size_per_image=
        num_proposal_samples_per_batch)
    self.net.transform = self.IdentityTransform()
