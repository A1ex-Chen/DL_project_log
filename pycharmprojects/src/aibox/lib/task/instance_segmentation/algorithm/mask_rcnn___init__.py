def __init__(self, num_classes: int, image_min_side: int, image_max_side: int):
    super().__init__(num_classes, image_min_side, image_max_side)
    mask_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained
        =True, min_size=image_min_side, max_size=image_max_side)
    mask_rcnn.rpn.anchor_generator = AnchorGenerator(sizes=mask_rcnn.rpn.
        anchor_generator.sizes, aspect_ratios=mask_rcnn.rpn.
        anchor_generator.aspect_ratios)
    in_features = mask_rcnn.roi_heads.box_predictor.cls_score.in_features
    mask_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features,
        num_classes)
    in_features_mask = (mask_rcnn.roi_heads.mask_predictor.conv5_mask.
        in_channels)
    hidden_layer = 256
    mask_rcnn.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
        hidden_layer, num_classes)
    self.net = mask_rcnn
