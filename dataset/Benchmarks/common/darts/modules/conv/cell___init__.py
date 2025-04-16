def __init__(self, num_nodes, multiplier, cpp, cp, c, reduction, reduction_prev
    ):
    """
        :param steps: 4, number of layers inside a cell
        :param multiplier: 4
        :param cpp: 48
        :param cp: 48
        :param c: 16
        :param reduction: indicates whether to reduce the output maps width
        :param reduction_prev: when previous cell reduced width, s1_d = s0_d//2
        in order to keep same shape between s1 and s0, we adopt prep0 layer to
        reduce the s0 width by half.
        """
    super(Cell, self).__init__()
    self.reduction = reduction
    self.reduction_prev = reduction_prev
    if reduction_prev:
        self.preprocess0 = FactorizedReduce(cpp, c, affine=False)
    else:
        self.preprocess0 = ConvBlock(cpp, c, 1, 1, 0, affine=False)
    self.preprocess1 = ConvBlock(cp, c, 1, 1, 0, affine=False)
    self.num_nodes = num_nodes
    self.multiplier = multiplier
    self.layers = nn.ModuleList()
    for i in range(self.num_nodes):
        for j in range(2 + i):
            stride = 2 if reduction and j < 2 else 1
            layer = MixedLayer(c, stride)
            self.layers.append(layer)
