def batchnorm(self, planes, zero_init=False):
    bn_cfg = {}
    if self.config.bn_momentum is not None:
        bn_cfg['momentum'] = self.config.bn_momentum
    if self.config.bn_epsilon is not None:
        bn_cfg['eps'] = self.config.bn_epsilon
    bn = nn.BatchNorm2d(planes, **bn_cfg)
    gamma_init_val = 0 if zero_init else 1
    nn.init.constant_(bn.weight, gamma_init_val)
    nn.init.constant_(bn.bias, 0)
    return bn
