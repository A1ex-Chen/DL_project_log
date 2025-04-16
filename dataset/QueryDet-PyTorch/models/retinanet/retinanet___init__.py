def __init__(self, cfg, input_shape: List[ShapeSpec]):
    super().__init__()
    in_channels = input_shape[0].channels
    num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
    num_convs = cfg.MODEL.RETINANET.NUM_CONVS
    prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
    num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
    assert len(set(num_anchors)
        ) == 1, 'Using different number of anchors between levels is not currently supported!'
    num_anchors = num_anchors[0]
    cls_subnet = []
    bbox_subnet = []
    for _ in range(num_convs):
        cls_subnet.append(nn.Conv2d(in_channels, in_channels, kernel_size=3,
            stride=1, padding=1))
        cls_subnet.append(nn.ReLU())
        bbox_subnet.append(nn.Conv2d(in_channels, in_channels, kernel_size=
            3, stride=1, padding=1))
        bbox_subnet.append(nn.ReLU())
    self.cls_subnet = nn.Sequential(*cls_subnet)
    self.bbox_subnet = nn.Sequential(*bbox_subnet)
    self.cls_score = nn.Conv2d(in_channels, num_anchors * num_classes,
        kernel_size=3, stride=1, padding=1)
    self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3,
        stride=1, padding=1)
    for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self
        .bbox_pred]:
        for layer in modules.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)
    bias_value = -math.log((1 - prior_prob) / prior_prob)
    torch.nn.init.constant_(self.cls_score.bias, bias_value)
