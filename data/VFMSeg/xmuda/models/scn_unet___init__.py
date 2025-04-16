def __init__(self, in_channels, m=16, block_reps=1, residual_blocks=False,
    full_scale=4096, num_planes=7):
    super(UNetSCN, self).__init__()
    self.in_channels = in_channels
    self.out_channels = m
    n_planes = [((n + 1) * m) for n in range(num_planes)]
    self.sparseModel = scn.Sequential().add(scn.InputLayer(DIMENSION,
        full_scale, mode=4)).add(scn.SubmanifoldConvolution(DIMENSION,
        in_channels, m, 3, False)).add(scn.UNet(DIMENSION, block_reps,
        n_planes, residual_blocks)).add(scn.BatchNormReLU(m)).add(scn.
        OutputLayer(DIMENSION))
