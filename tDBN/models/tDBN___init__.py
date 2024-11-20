def __init__(self, output_shape, use_norm=True, num_filters_down1=[32, 64, 
    96, 128], num_filters_down2=[32, 64, 96, 128], name='tDBN_bv_2'):
    super(tDBN_bv_2, self).__init__()
    self.name = name
    if use_norm:
        BatchNorm1d = change_default_args(eps=0.001, momentum=0.01)(nn.
            BatchNorm1d)
        Linear = change_default_args(bias=False)(nn.Linear)
    else:
        BatchNorm1d = Empty
        Linear = change_default_args(bias=True)(nn.Linear)
    sparse_shape = np.array(output_shape[1:4])
    self.scn_input = scn.InputLayer(3, sparse_shape.tolist())
    self.voxel_output_shape = output_shape
    To_use_bias = False
    residual_use = True
    dimension = 3
    reps = 2
    dimension = 3
    leakiness = 0
    input_filters_layers = num_filters_down1[:4]
    num_filter_fpn = num_filters_down1[3:]
    dimension_feature_map = num_filters_down2
    dimension_kernel_size = [15, 7, 3, 1]
    filters_input_pairs = [[input_filters_layers[i], input_filters_layers[i +
        1]] for i in range(len(input_filters_layers) - 1)]
    m = None
    m = scn.Sequential()
    for i, o in [[1, input_filters_layers[0]]]:
        m.add(scn.SubmanifoldConvolution(3, i, o, 3, False))
    for i, o in filters_input_pairs:
        for _ in range(reps):
            self.block(m, i, i, residual_blocks=residual_use)
        m.add(scn.BatchNormLeakyReLU(i, leakiness=leakiness)).add(scn.
            Convolution(dimension, i, o, 3, 2, False))
    self.block_input = m
    middle_layers = []
    m = None
    m = scn.Sequential()
    for _ in range(reps):
        self.block(m, num_filter_fpn[0], num_filter_fpn[0], residual_blocks
            =residual_use)
    self.x0_in = m
    for k in range(1, 4):
        m = None
        m = scn.Sequential()
        m.add(scn.BatchNormLeakyReLU(num_filter_fpn[k - 1], leakiness=
            leakiness)).add(scn.Convolution(dimension, num_filter_fpn[k - 1
            ], num_filter_fpn[k], 3, 2, False))
        for _ in range(reps):
            if k == 4:
                self.block(m, num_filter_fpn[k], num_filter_fpn[k],
                    dimension=2, residual_blocks=residual_use)
            else:
                self.block(m, num_filter_fpn[k], num_filter_fpn[k],
                    dimension=3, residual_blocks=residual_use)
        if k == 1:
            self.x1_in = m
        if k == 2:
            self.x2_in = m
        if k == 3:
            self.x3_in = m
    self.feature_map3 = scn.Sequential(scn.BatchNormLeakyReLU(
        num_filter_fpn[3], leakiness=leakiness)).add(scn.SparseToDense(3,
        num_filter_fpn[3]))
    for k in range(2, -1, -1):
        m = None
        m = scn.Sequential()
        m.add(scn.BatchNormLeakyReLU(num_filter_fpn[k + 1], leakiness=
            leakiness)).add(scn.Deconvolution(dimension, num_filter_fpn[k +
            1], num_filter_fpn[k], 3, 2, False))
        if k == 2:
            self.upsample32 = m
        if k == 1:
            self.upsample21 = m
        if k == 0:
            self.upsample10 = m
        m = None
        m = scn.Sequential()
        m.add(scn.JoinTable())
        for i in range(reps):
            self.block(m, num_filter_fpn[k] * (2 if i == 0 else 1),
                num_filter_fpn[k], residual_blocks=residual_use)
        if k == 2:
            self.concate2 = m
        if k == 1:
            self.concate1 = m
        if k == 0:
            self.concate0 = m
        m = None
        m = scn.Sequential()
        m.add(scn.BatchNormLeakyReLU(num_filter_fpn[k], leakiness=leakiness)
            ).add(scn.Convolution(3, num_filter_fpn[k],
            dimension_feature_map[k], (dimension_kernel_size[k], 1, 1), (1,
            1, 1), bias=False)).add(scn.BatchNormReLU(dimension_feature_map
            [k], eps=0.001, momentum=0.99)).add(scn.SparseToDense(3,
            dimension_feature_map[k]))
        if k == 2:
            self.feature_map2 = m
        if k == 1:
            self.feature_map1 = m
        if k == 0:
            self.feature_map0 = m
