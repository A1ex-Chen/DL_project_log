def __init__(self, use_norm=True, num_class=2, layer_nums=[3, 3, 3, 3],
    layer_strides=[2, 2, 2, 2], num_filters=[64, 128, 256, 512],
    upsample_strides=[1, 2, 4, 4], num_upsample_filters=[64, 128, 256, 256,
    448], num_input_filters=128, num_anchor_per_loc=2,
    encode_background_as_zeros=True, use_direction_classifier=True,
    use_groupnorm=False, num_groups=32, use_bev=False, box_code_size=7,
    name='det_net_2'):
    super(det_net_2, self).__init__()
    self._num_anchor_per_loc = num_anchor_per_loc
    self._use_direction_classifier = use_direction_classifier
    self._use_bev = use_bev
    assert len(layer_nums) == 4
    assert len(layer_strides) == len(layer_nums)
    assert len(num_filters) == len(layer_nums)
    assert len(upsample_strides) == len(layer_nums)
    if use_norm:
        if use_groupnorm:
            BatchNorm2d = change_default_args(num_groups=num_groups, eps=0.001
                )(GroupNorm)
        else:
            BatchNorm2d = change_default_args(eps=0.001, momentum=0.01)(nn.
                BatchNorm2d)
        Conv2d = change_default_args(bias=False)(nn.Conv2d)
        ConvTranspose2d = change_default_args(bias=False)(nn.ConvTranspose2d)
    else:
        BatchNorm2d = Empty
        Conv2d = change_default_args(bias=True)(nn.Conv2d)
        ConvTranspose2d = change_default_args(bias=True)(nn.ConvTranspose2d)
    dimension_feature_map = num_filters
    dimension_concate = num_upsample_filters
    flag = 0
    middle_layers = []
    for i in range(layer_nums[flag]):
        middle_layers.append(Conv2d(dimension_feature_map[flag],
            dimension_feature_map[flag], 3, padding=1))
        middle_layers.append(BatchNorm2d(dimension_feature_map[flag]))
        middle_layers.append(nn.ReLU())
    self.block0 = Sequential(*middle_layers)
    middle_layers = []
    self.downsample0 = Sequential(Conv2d(dimension_feature_map[flag],
        dimension_concate[flag], 3, stride=2), BatchNorm2d(
        dimension_concate[flag]), nn.ReLU())
    flag = 1
    middle_layers = []
    for i in range(layer_nums[flag]):
        middle_layers.append(Conv2d(dimension_feature_map[flag],
            dimension_feature_map[flag], 3, padding=1))
        middle_layers.append(BatchNorm2d(dimension_feature_map[flag]))
        middle_layers.append(nn.ReLU())
    self.block1 = Sequential(*middle_layers)
    flag = 2
    middle_layers = []
    for i in range(layer_nums[flag]):
        middle_layers.append(Conv2d(dimension_feature_map[flag],
            dimension_feature_map[flag], 3, padding=1))
        middle_layers.append(BatchNorm2d(dimension_feature_map[flag]))
        middle_layers.append(nn.ReLU())
    self.block2 = Sequential(*middle_layers)
    self.upsample2 = Sequential(ConvTranspose2d(dimension_feature_map[flag],
        dimension_concate[flag], 3, stride=2), BatchNorm2d(
        dimension_concate[flag]), nn.ReLU())
    flag = 3
    middle_layers = []
    for i in range(layer_nums[flag]):
        middle_layers.append(Conv2d(dimension_feature_map[flag],
            dimension_feature_map[flag], 3, padding=1))
        middle_layers.append(BatchNorm2d(dimension_feature_map[flag]))
        middle_layers.append(nn.ReLU())
    self.block3 = Sequential(*middle_layers)
    self.upsample3 = Sequential(ConvTranspose2d(dimension_feature_map[flag],
        dimension_concate[flag], 5, stride=4, output_padding=2),
        BatchNorm2d(dimension_concate[flag]), nn.ReLU())
    middle_layers = []
    middle_layers.append(Conv2d(dimension_concate[0] +
        dimension_feature_map[1] + dimension_concate[2] + dimension_concate
        [3], dimension_concate[4], 3, padding=1))
    middle_layers.append(BatchNorm2d(dimension_concate[4]))
    middle_layers.append(nn.ReLU())
    middle_layers.append(Conv2d(dimension_concate[4], dimension_concate[4],
        3, padding=1))
    middle_layers.append(BatchNorm2d(dimension_concate[4]))
    middle_layers.append(nn.ReLU())
    self.output_after_concate_fuse3210 = Sequential(*middle_layers)
    if encode_background_as_zeros:
        num_cls = num_anchor_per_loc * num_class
    else:
        num_cls = num_anchor_per_loc * (num_class + 1)
    self.conv_cls = nn.Conv2d(dimension_concate[4], num_cls, 1)
    self.conv_box = nn.Conv2d(dimension_concate[4], num_anchor_per_loc *
        box_code_size, 1)
    if use_direction_classifier:
        self.conv_dir_cls = nn.Conv2d(dimension_concate[4], 
            num_anchor_per_loc * 2, 1)
