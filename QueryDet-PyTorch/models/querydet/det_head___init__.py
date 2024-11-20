def __init__(self, in_channels, conv_channels, num_convs, pred_channels,
    pred_prior=None):
    super().__init__()
    self.num_convs = num_convs
    self.bn_converted = False
    self.subnet = []
    self.bns = []
    channels = in_channels
    for i in range(self.num_convs):
        layer = Conv2d(channels, conv_channels, kernel_size=3, stride=1,
            padding=1, bias=False, activation=None, norm=None)
        torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
        bn = MergedSyncBatchNorm(conv_channels)
        self.add_module('layer_{}'.format(i), layer)
        self.add_module('bn_{}'.format(i), bn)
        self.subnet.append(layer)
        self.bns.append(bn)
        channels = conv_channels
    self.pred_net = nn.Conv2d(channels, pred_channels, kernel_size=3,
        stride=1, padding=1)
    torch.nn.init.normal_(self.pred_net.weight, mean=0, std=0.01)
    if pred_prior is not None:
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.pred_net.bias, bias_value)
    else:
        torch.nn.init.constant_(self.pred_net.bias, 0)
