def __init__(self, num_classes, anchors=1, in_channels=(128, 128, 128, 128,
    128, 128), inplace=True, prior_prob=0.01):
    super().__init__()
    if isinstance(anchors, (list, tuple)):
        self.n_anchors = len(anchors)
    else:
        self.n_anchors = anchors
    self.num_classes = num_classes
    self.cls_preds = nn.ModuleList()
    self.reg_preds = nn.ModuleList()
    self.obj_preds = nn.ModuleList()
    cls_in_channels = in_channels[0::2]
    reg_in_channels = in_channels[1::2]
    for cls_in_channel, reg_in_channel in zip(cls_in_channels, reg_in_channels
        ):
        cls_pred = nn.Conv2d(in_channels=cls_in_channel, out_channels=self.
            n_anchors * self.num_classes, kernel_size=1, stride=1, padding=0)
        reg_pred = nn.Conv2d(in_channels=reg_in_channel, out_channels=4,
            kernel_size=1, stride=1, padding=0)
        obj_pred = nn.Conv2d(in_channels=reg_in_channel, out_channels=self.
            n_anchors * 1, kernel_size=1, stride=1, padding=0)
        self.cls_preds.append(cls_pred)
        self.reg_preds.append(reg_pred)
        self.obj_preds.append(obj_pred)
    self.nc = self.num_classes
    self.nl = len(cls_in_channels)
    self.na = self.n_anchors
    self.use_l1 = False
    self.l1_loss = nn.L1Loss(reduction='none')
    self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction='none')
    self.iou_loss = IOUloss(reduction='none')
    self.grids = [torch.zeros(1)] * len(in_channels)
    self.xy_shifts = [torch.zeros(1)] * len(in_channels)
    self.org_grids = [torch.zeros(1)] * len(in_channels)
    self.grid_sizes = [[0, 0, 0] for _ in range(len(in_channels))]
    self.expanded_strides = [torch.zeros(1)] * len(in_channels)
    self.center_ltrbes = [torch.zeros(1)] * len(in_channels)
    self.center_radius = 2.5
    self.prior_prob = prior_prob
    self.inplace = inplace
